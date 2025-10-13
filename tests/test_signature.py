import time
import uuid
import hmac
import hashlib
import base64

# 1. 定义用户信息和消息
user_id = "7db50472-f33b-4041-9b17-543704686133"
user_message = "接下来呢？"

# 2. 生成签名所需参数
timestamp = 1760341685487
request_id = 'dfb576f5-a33c-4a93-8b47-b1a81df3ae27'
# chat_id = str(uuid.uuid4())

# 3. 构造签名
safe_user_message = user_message or ""
e = f"requestId,{request_id},timestamp,{timestamp},user_id,{user_id}"
msg_encode = base64.b64encode(safe_user_message.encode('utf-8')).decode('utf-8')
i = f"{e}|{msg_encode}|{str(timestamp)}"
n = timestamp // (5 * 60 * 1000)
key = "junjie".encode('utf-8')
o = hmac.new(key, str(n).encode('utf-8'), hashlib.sha256).hexdigest()
print(o)
print(o == "23d5377f55d1e213639b1533f6f6116724960aad0447497b71965146c6d110d8")

signature = hmac.new(o.encode('utf-8'), i.encode('utf-8'), hashlib.sha256).hexdigest()

print(signature)
print(signature == "fd9acecbeb66030f866f49e90a73a24622fc2f65a4980ab0710ef78377f1f855")

'requestId,dfb576f5-a33c-4a93-8b47-b1a81df3ae27,timestamp,1760341685487,user_id,7db50472-f33b-4041-9b17-543704686133|接下来呢？|1760341685487'
"requestId,dfb576f5-a33c-4a93-8b47-b1a81df3ae27,timestamp,1760341685487,user_id,7db50472-f33b-4041-9b17-543704686133|5o6l5LiL5p2l5ZGi77yf|1760341685487"
