class Dict(dict):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)        
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_nested_dict(cls, content:dict):
        content = cls(**content)
        for k, v in content.items():
            if isinstance(v, dict):
                content[k] = cls(**v)

        for k, v in content.items():
            if isinstance(v, dict):
                setattr(content, k, cls(**v))
        return content
