---
id: 84301e35dd
question: ConnectionError 'Connection aborted.' for --bind 127.0.0.1:5000
sort_order: 28
---

I was getting an error on the client side with this:

**Client Side Error:**

```plaintext
File "C:\python\lib\site-packages\urllib3\connectionpool.py", line 703, in urlopen ...

raise ConnectionError(err, request=request)

requests.exceptions.ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))
```

**Server Side:**

An error was shown for Gunicorn, although the `waitress` command was running smoothly from the server side.

**Solution:**

- Use the IP address `0.0.0.0:8000` or `0.0.0.0:9696`. They are the ones which work most of the time.