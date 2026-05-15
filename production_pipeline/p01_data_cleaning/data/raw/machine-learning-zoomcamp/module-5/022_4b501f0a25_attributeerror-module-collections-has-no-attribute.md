---
id: 4b501f0a25
question: 'AttributeError: module ‘collections’ has no attribute ‘MutableMapping’'
sort_order: 22
---

Following the instruction from video week-5.6, using pipenv to install Python libraries throws the error shown below:

```
naneen@xps:ml_zoomcamp_ht$ pipenv install numpy
Traceback (most recent call last):
  File "/usr/bin/pipenv", line 33, in <module>
    sys.exit(load_entry_point('pipenv==11.9.0', 'console_scripts', 'pipenv')())
  File "/usr/lib/python3/dist-packages/pipenv/vendor/click/core.py", line 722, in __call__
    return self.main(*args, **kwargs)
  File "/usr/lib/python3/dist-packages/pipenv/vendor/click/core.py", line 697, in main
    rv = self.invoke(ctx)
  File "/usr/lib/python3/dist-packages/pipenv/vendor/click/core.py", line 1066, in invoke
    return _process_result(sub_ctx.command.invoke(sub_ctx))
  File "/usr/lib/python3/dist-packages/pipenv/vendor/click/core.py", line 895, in invoke
    return ctx.invoke(self.callback, **ctx.params)
  File "/usr/lib/python3/dist-packages/pipenv/vendor/click/core.py", line 535, in invoke
    return callback(*args, **kwargs)
  File "/usr/lib/python3/dist-packages/pipenv/cli/py.py", line 347, in install
    from . import core
  File "/usr/lib/python3/dist-packages/pipenv/core.py", line 21, in <module>
    import requests
  File "/usr/lib/python3/dist-packages/pipenv/vendor/requests/__init__.py", line 65, in <module>
    from . import utils
  File "/usr/lib/python3/dist-packages/pipenv/vendor/requests/utils.py", line 27, in <module>
    from .cookies import RequestsCookieJar, cookiejar_from_dict
  File "/usr/lib/python3/dist-packages/pipenv/vendor/requests/cookies.py", line 172, in <module>
    class RequestsCookieJar(cookielib.CookieJar, collections.MutableMapping):
AttributeError: module 'collections' has no attribute 'MutableMapping'
naneen@xps:ml_zoomcamp_ht$
```

Solution:

- Ensure you are working with Python version 3.10+