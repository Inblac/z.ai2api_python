import time
import uuid
import hmac
import hashlib
import base64

# 1. 定义用户信息和消息
user_id = "7db50472-f33b-4041-9b17-543704686133"
user_message = "一下降了10度"

# 2. 生成签名所需参数
timestamp = 1760715902344
request_id = '589d5f4a-1409-49af-b3e1-246c44810905'
# chat_id = str(uuid.uuid4())

# 3. 构造签名
safe_user_message = user_message or ""
e = f"requestId,{request_id},timestamp,{timestamp},user_id,{user_id}"
msg_encode = base64.b64encode(safe_user_message.encode('utf-8')).decode('utf-8')
i = f"{e}|{msg_encode}|{str(timestamp)}"
n = timestamp // (5 * 60 * 1000)
key = "key-@@@@)))()((9))-xxxx&&&%%%%%".encode('utf-8')
o = hmac.new(key, str(n).encode('utf-8'), hashlib.sha256).hexdigest()
# print(o)
# print(o == "23d5377f55d1e213639b1533f6f6116724960aad0447497b71965146c6d110d8")

signature = hmac.new(o.encode('utf-8'), i.encode('utf-8'), hashlib.sha256).hexdigest()

print(signature)
print(signature == "72e53c9a764cd2a20def415b9dcd7a49f20fe9ee5fe0ba1c1108cdbc0d4bf409")

'requestId,dfb576f5-a33c-4a93-8b47-b1a81df3ae27,timestamp,1760341685487,user_id,7db50472-f33b-4041-9b17-543704686133|接下来呢？|1760341685487'
"requestId,5e591161-df1d-4aa2-b083-d81484379539,timestamp,1760450591302,user_id,7db50472-f33b-4041-9b17-543704686133|bmloYW8=|1760450591302"

"requestId,589d5f4a-1409-49af-b3e1-246c44810905,timestamp,1760715902344,user_id,7db50472-f33b-4041-9b17-543704686133|5LiA5LiL6ZmN5LqGMTDluqY=|1760715902344"