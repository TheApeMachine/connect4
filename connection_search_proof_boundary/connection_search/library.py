"""Reusable closed functions, not opaque capabilities or learned model weights.
`use` is transparent macro expansion to an ordinary fn term before checking.
"""
from __future__ import annotations
import copy
import json
from pathlib import Path
from .dsl import Invalid,children,digest,fn,free_vars,holes,infer,key,size,validate_shape,walk_typed
from .algebra import scan_laws


class Library:
    def __init__(self,entries=None):
        self.entries={}
        for entry in entries or []:
            term=entry['term']
            validate_shape(term)
            if free_vars(term) or holes(term):
                raise Invalid('Library entries must be closed, hole-free functions')
            typ=infer(term,{})
            if not isinstance(typ,list) or typ[0]!='Fn':
                raise Invalid('Library entry must be a function')
            scan_laws(term,{})
            eid='lib_'+digest(term)
            if entry.get('id')!=eid or entry.get('type')!=typ:
                raise Invalid('Library identity/type does not match its transparent body')
            # Imported evidence is provenance, not trusted correctness.
            self.entries[eid]=copy.deepcopy(entry)

    @classmethod
    def load(cls,path):
        if not path:
            return cls()
        data=json.loads(Path(path).read_text())
        if not isinstance(data,dict) or data.get('schema')!=1:
            raise ValueError('Unsupported library format')
        return cls(data['entries'])

    @property
    def fingerprint(self):
        return digest(self.to_dict())

    def clone(self):
        return Library(list(self.entries.values()))

    def resolve(self,expr):
        def walk(x,depth=0):
            if depth>40:
                raise Invalid('Macro/expression depth exceeded')
            if isinstance(x,list) and len(x)==2 and x[0]=='use':
                if not isinstance(x[1],str) or x[1] not in self.entries:
                    raise Invalid('Unknown reusable function id')
                return copy.deepcopy(self.entries[x[1]]['term'])
            if isinstance(x,list):
                # Metadata (parameter/type arrays) contains no executable use forms.
                if x and x[0]=='fn' and len(x)==3:
                    return ['fn',copy.deepcopy(x[1]),walk(x[2],depth+1)]
                if x and x[0]=='hole':
                    return copy.deepcopy(x)
                return [x[0]]+[walk(c,depth+1) for c in x[1:]] if x else []
            return x
        out=walk(expr)
        validate_shape(out)
        return out

    def harvest(self,expr,env,source,check,max_entries=8):
        if not check.passed:
            raise ValueError('Only search-tested solutions can supply reusable compositions')
        candidates=[]
        for path,term,local,t in walk_typed(expr,env):
            if size(term)<3 or holes(term):
                continue
            names=sorted(free_vars(term))
            if len(names)>4:
                continue
            if names:
                closed=['fn',[[n,local[n]] for n in names],term]
            elif isinstance(t,list) and t[0]=='Fn':
                closed=term
            else:
                continue
            candidates.append((size(term),path,closed))
        added=[]
        for _,path,closed in sorted(candidates,reverse=True):
            eid='lib_'+digest(closed)
            if eid in self.entries:
                continue
            validate_shape(closed)
            typ=infer(closed,{})
            try:
                laws=scan_laws(closed,{})
            except Invalid:
                continue
            entry={'id':eid,'term':closed,'type':typ,'source':source,'source_path':list(path),
                   'evidence':{'type_checked':True,'source_program_status':check.status,
                               'subexpression_has_independent_goal_proof':False},
                   'law_certificates':laws}
            self.entries[eid]=entry
            added.append(eid)
            if len(added)>=max_entries:
                break
        return added

    def offered(self,limit=8):
        return [self.entries[k] for k in sorted(self.entries)[:limit]]

    def to_dict(self):
        return {'schema':1,'entries':[self.entries[k] for k in sorted(self.entries)]}

    def save(self,path):
        Path(path).write_text(json.dumps(self.to_dict(),indent=2)+'\n')
