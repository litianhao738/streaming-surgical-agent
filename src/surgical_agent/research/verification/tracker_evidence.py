"""Bounded predicted-track hints for post-Gate visual review, never GT or decisions."""
from copy import deepcopy
import json
from surgical_agent.tracking.contracts import PredictedTrack
from surgical_agent.research.verification.candidate_coordinator import _TASK_NAMES

VERSION='post_gate_tracker_evidence_v1'
GUIDANCE=(
    'The attached tracks are fallible detector/association predictions, not annotations or another reviewer vote. '
    'Use boxes only to locate tools in the original images and IDs only as tentative cross-image correspondences. '
    'Check the pixels before accepting any class or identity. A class or trajectory alone does not establish a verb, '
    'contacted anatomy, or IVT. Missing detections and local boxes cannot refute a whole-frame candidate. '
    'History-only evidence is not current evidence. Judge only the existing propositions; preserve the original '
    'rating scale, uncertainty rules and response_schema. Do not add candidates or copy detector confidence as a rating.'
)

def make_packet(snapshot,selected,max_tracks=6):
    if selected.get('source_split')!='Training':raise ValueError('Training-only evidence')
    frames=selected['causal_frame_ids']
    if len(frames)!=3 or sorted(set(frames))!=list(frames) or frames[-1]!=selected['frame_id']:
        raise ValueError('invalid causal window')
    if snapshot.get('status')!='AVAILABLE' or snapshot.get('video_id')!=selected['video_id']:
        raise ValueError('Tracker status/video mismatch')
    if snapshot.get('source_max_frame_id')!=selected['frame_id']:
        raise ValueError('Tracker future or stale evidence')
    if [f['frame_id'] for f in snapshot['frames']]!=list(frames):raise ValueError('Tracker frame alignment mismatch')
    if not isinstance(max_tracks,int) or not 1<=max_tracks<=6:raise ValueError('evidence bound exceeded')
    ids={};out=[]
    for image_index,f in enumerate(snapshot['frames']):
        tracks=[]
        for t in f['tracks']:
            # Revalidate numeric boxes/classes; never forward arbitrary text/metadata.
            obj=PredictedTrack(t['track_id'],t['instrument_id'],tuple(t['bbox_tlwh']),t['score'],t['age'])
            tracks.append(obj)
        tracks=sorted(tracks,key=lambda t:(-t.score,t.track_id))[:max_tracks]
        hints=[]
        for t in tracks:
            if t.track_id not in ids:ids[t.track_id]='T'+str(len(ids)+1)
            hints.append({'track':ids[t.track_id],'instrument_hypothesis':_TASK_NAMES['instrument'][t.instrument_id],
                'instrument_id':t.instrument_id,'confidence':round(t.score,3),
                'bbox_tlwh':[round(v,5) for v in t.bbox_tlwh]})
        out.append({'image_index':image_index,'tracks':hints})
    packet={'version':VERSION,'coordinate_system':'normalized top-left x,y,width,height in the original full image',
        'current_image_index':2,'maximum_tracks_per_image':max_tracks,'frames':out}
    if len(json.dumps(packet).encode())>10000:raise ValueError('evidence packet too large')
    return packet

def attach(body,packet):
    """Only two additive prompt fields; exact same images, candidates and output schema."""
    out=deepcopy(body);content=out['messages'][0]['content']
    if len(out['messages'])!=1 or content[0]['type']!='text':raise ValueError('unknown review wire')
    if sum(b.get('type')=='image_url' for b in content)!=3:raise ValueError('three original images required')
    prompt=json.loads(content[0]['text'])
    if 'propositions' not in prompt or 'response_schema' not in prompt:raise ValueError('compact review required')
    if 'tracker_evidence' in prompt:raise ValueError('duplicate Tracker attachment')
    prompt['tracker_evidence']=deepcopy(packet);prompt['tracker_evidence_instructions']=GUIDANCE
    content[0]['text']=json.dumps(prompt,ensure_ascii=False)
    return out
