"""Durable shared per-dispatch budget, separate from per-target response journals."""
import json
import sqlite3
from decimal import Decimal
from threading import RLock


class BudgetStop(RuntimeError):
    pass


class AmbiguousDispatch(RuntimeError):
    pass


class Budget:
    def __init__(self, path, limits, plan_hash):
        self.lock = RLock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        with self.db:
            self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
            self.db.execute("CREATE TABLE IF NOT EXISTS accounts (name TEXT PRIMARY KEY, cap TEXT, occupied TEXT)")
            self.db.execute("CREATE TABLE IF NOT EXISTS calls (key TEXT PRIMARY KEY, account TEXT, charge TEXT, state TEXT)")
            old = self.db.execute("SELECT value FROM meta WHERE key='plan'").fetchone()
            if old and old[0] != plan_hash:
                raise ValueError("budget belongs to a different frozen plan")
            self.db.execute("INSERT OR IGNORE INTO meta VALUES ('plan',?)", (plan_hash,))
            for account, cap in limits.items():
                self.db.execute("INSERT OR IGNORE INTO accounts VALUES (?,?, '0')", (account,str(cap)))
                if self.db.execute("SELECT cap FROM accounts WHERE name=?",(account,)).fetchone()[0] != str(cap):
                    raise ValueError("budget cap changed")

    @staticmethod
    def key(target, stage, seat):
        return json.dumps([target,stage,seat],separators=(',',':'))

    def reserve(self, key, account, amount):
        if not amount.is_finite() or amount < 0:
            raise ValueError('invalid reservation amount')
        with self.lock, self.db:
            if self.db.execute("SELECT 1 FROM calls WHERE key=?",(key,)).fetchone():
                raise AmbiguousDispatch("existing dispatch cannot be submitted again: "+key)
            cap, occupied = map(Decimal,self.db.execute("SELECT cap,occupied FROM accounts WHERE name=?",(account,)).fetchone())
            if occupied+amount > cap:
                raise BudgetStop("budget reservation exceeds "+account)
            self.db.execute("INSERT INTO calls VALUES (?,?,?,'RESERVED')",(key,account,str(amount)))
            self.db.execute("UPDATE accounts SET occupied=? WHERE name=?",(str(occupied+amount),account))

    def settle(self, key, charge):
        if not charge.is_finite() or charge < 0:
            raise ValueError('invalid provider charge')
        with self.lock, self.db:
            found = self.db.execute("SELECT account,charge FROM calls WHERE key=?",(key,)).fetchone()
            if found is None:
                raise ValueError("unreserved dispatch")
            account, old = found
            occupied = Decimal(self.db.execute("SELECT occupied FROM accounts WHERE name=?",(account,)).fetchone()[0])
            self.db.execute("UPDATE accounts SET occupied=? WHERE name=?",(str(occupied+charge-Decimal(old)),account))
            self.db.execute("UPDATE calls SET charge=?,state='TERMINAL' WHERE key=?",(str(charge),key))

    def pending(self):
        with self.lock:
            return [json.loads(r[0]) for r in self.db.execute("SELECT key FROM calls WHERE state='RESERVED'")]

    def summary(self):
        with self.lock:
            return {"accounts": {r[0]:{"cap":r[1],"occupied":r[2]} for r in self.db.execute("SELECT * FROM accounts")},
                    "dispatches": self.db.execute("SELECT COUNT(*) FROM calls").fetchone()[0],
                    "pending":len(self.pending())}

    def close(self):
        self.db.close()
