"""Content-addressed typed hypergraph; immutable alternatives share subexpressions."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from .dsl import at,children,digest,free_vars,infer,key,walk_typed


class Graph:
    def __init__(self):
        self.nodes={}
        self.edges={}
        self.roots=[]
        self.revisions=[]

    def record(self,expr,env,check,parent=None,focus=()):
        typed=list(walk_typed(expr,env))
        ids={}
        for path,e,local,t in reversed(typed):
            relevant={v:local[v] for v in sorted(free_vars(e)) if v in local}
            nid=digest([e,relevant])
            ids[path]=nid
            if nid not in self.nodes:
                self.nodes[nid]={'id':nid,'expression':e,'type':t,'free_variables':relevant,
                                 'status':'type_checked','assessments':[]}
            tail=[ids[path+(i,)] for i,_ in children(e)]
            if tail:
                op=e[0]
                self.edges[nid]={'id':'e_'+nid,'head':nid,'tails':tail,'operator':op,
                                'semantics':'all tails jointly required; ordered arguments',
                                'bindings':e[1] if op=='fn' else []}
        root=ids[()]
        self.nodes[root]['assessments'].append(check.to_dict())
        if root not in self.roots:
            self.roots.append(root)
        if parent is not None:
            self.revisions.append({'parent_root':parent,'new_root':root,'replaced_path':list(focus),
                                   'semantics':'immutable fork; dependent ancestors rebuilt, other nodes retained'})
        for proof in check.certificates:
            function_path=tuple(proof['path'])+(1,)
            if function_path in ids:
                node=self.nodes[ids[function_path]]
                node.setdefault('properties',{})['associativity']=proof
        return root

    def to_dict(self):
        return {'schema':1,'nodes':list(self.nodes.values()),'hyperedges':list(self.edges.values()),
                'roots':self.roots,'revisions':self.revisions}

    def dot(self):
        lines=['digraph search {','rankdir=LR;']
        for n in self.nodes.values():
            e=n['expression']
            label=(e[0] if isinstance(e,list) else str(e))+'\\n'+n['id'][:8]
            lines.append('"'+n['id']+'" [label='+json.dumps(label)+'];')
        for e in self.edges.values():
            # Explicit conjunction vertex: separate tails do not each imply head.
            lines.append('"'+e['id']+'" [shape=point];')
            for i,t in enumerate(e['tails']):
                lines.append(f'"{t}" -> "{e["id"]}" [label="{i}"];')
            lines.append(f'"{e["id"]}" -> "{e["head"]}";')
        lines.append('}')
        return '\n'.join(lines)+'\n'

    def available(self,expected,env,exclude=None,limit=8,*,functions_only=False):
        from .dsl import holes,size
        candidates=[]
        for n in self.nodes.values():
            e=n['expression']
            if ((expected is not None and n['type']!=expected) or holes(e)
                    or (exclude is not None and e==exclude)):
                continue
            if functions_only and not (isinstance(n['type'],list) and n['type'][0]=='Fn'):
                continue
            if not all(env.get(v)==t for v,t in n['free_variables'].items()):
                continue
            candidates.append(n)
        candidates.sort(key=lambda n:(size(n['expression']),n['id']))
        return [{'node_id':n['id'],'term':n['expression'],'type':n['type'],
                 'evidence':'type_checked; no independent claim of task correctness',
                 'properties':list(n.get('properties',{}))} for n in candidates[:limit]]
