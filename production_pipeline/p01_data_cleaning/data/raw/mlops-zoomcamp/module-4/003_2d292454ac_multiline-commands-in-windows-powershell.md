---
id: 2d292454ac
question: Multiline commands in Windows Powershell
sort_order: 3
---

To use multiline commands in Windows PowerShell, place a backtick (`) at the end of each line except the last. Note that multiline strings do not require a backtick.

- Escape double quotes (`"`) to `"\`
- Use `$env:` to create environment variables (non-persistent). For example:

```powershell
$env:KINESIS_STREAM_INPUT="ride_events"

aws kinesis put-record --cli-binary-format raw-in-base64-out `

--stream-name $env:KINESIS_STREAM_INPUT `

--partition-key 1 `

--data '{

\"ride\": {

\"PULocationID\": 130,

\"DOLocationID\": 205,

\"trip_distance\": 3.66

},

\"ride_id\": 156

}'
```