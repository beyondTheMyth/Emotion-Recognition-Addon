__module_name__ = "emotionsend"
__module_version__ = "0.1.0"
__module_description__ = "Some script description"
 
import hexchat # HexChat IRC interface
import time
cnc = hexchat.find_context(channel='#THE2021FER')
emotion = 'none'
emotion_id = -1
last_id = -1
string = ''
while True:
    try:    
        with open("pathtoemotiondata/emotion.data", "r") as infile:
            string = infile.read()
    except IOError:
        pass
    if len(string) > 0:
        emotion, emotion_id = string.split()
        emotion_id = int(emotion_id)
    if emotion is not '':
        print(emotion)
        print(emotion_id)
        if last_id is not emotion_id:
            hexchat.command(f'say {emotion} {emotion_id}')
        #time.sleep(1)
        last_id = emotion_id

