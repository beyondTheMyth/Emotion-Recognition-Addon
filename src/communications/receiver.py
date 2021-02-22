# -*- coding: UTF-8 -*-

import hexchat

__module_name__ = "the2021fer"
__module_version__ = "1.0"
__module_description__ = "A plugin for sending and receiving texts from HexChat"


# get all users in channel and add them in a list
# lasttalk: Time they last talked
# realname: Real name.

users = hexchat.get_list("users")
userdict = {}

hookPrintEmotion = None
hookUpdateUsers = None
    
# set hook to get every message sent by any other user
# and update each user's emotion inside a dictionary
def handleMessage(word):
    nick = word[0]
    msg = word[1]
    if msg in ['positive', 'neutral', 'negative']:
        d1 = {nick: msg}
        userdict.update(d1)
    elif msg == 'clear':
        del userdict[str(nick)]

def getMessage(word, word_eol, userdata):
    # print("Message was sent by " + word[0] + " and said " + word[1])
    handleMessage(word)
    return hexchat.EAT_NONE

for e in ("Channel Message", "Channel Msg Hilight"): 
    hexchat.hook_print(e, getMessage)

# hook timer that updates the user list every second
def updateUsers(userdata):
    global users
    users = hexchat.get_list("users")
    return True

hookUpdateUsers = hexchat.hook_timer(1000, updateUsers)

# hook timer that prints emotions of all users every 5 seconds
def printOverallEmotions(userdata):
    
    positives = 0
    neutrals = 0
    negatives = 0
    no_emotion = len(users) - 1 - len(userdict)
    
    if (len(users)>1):
        for u in list(userdict.values()):
            print(u)
            if u == 'positive':
                positives += 1
            elif u == 'neutral':
                neutrals += 1
            elif u == 'negative':
                negatives += 1
            else:
                no_emotion += 1
                
        
        positives_rate = positives/(len(userdict)) * 100 if len(userdict)>0 else 0.00
        neutrals_rate = neutrals/(len(userdict)) * 100 if len(userdict)>0 else 0.00
        negatives_rate = negatives/(len(userdict)) * 100 if len(userdict)>0 else 0.00
        
        hexchat.command(f"say positives: {positives_rate}% --- neutrals: {neutrals_rate}% --- negatives: {negatives_rate}% --- no emotion detected: {no_emotion} people")

    return True

hookPrintEmotion = hexchat.hook_timer(5000, printOverallEmotions)

# Update Timer Hooks
def stop_print(word, word_eol, userdata):
    global hookPrintEmotion
    if hookPrintEmotion is not None:
        hexchat.unhook(hookPrintEmotion)
        hookPrintEmotion = None
    else:
        hookPrintEmotion = hexchat.hook_timer(5000, printOverallEmotions)
    return hexchat.EAT_ALL

def stop_update(word, word_eol, userdata):
    global hookUpdateUsers
    if hookUpdateUsers is not None:
        hexchat.unhook(hookUpdateUsers)
        hookUpdateUsers = None
    else:
        hookUpdateUsers = hexchat.hook_timer(5000, updateUsers)
    return hexchat.EAT_ALL

hexchat.hook_command("ST_PRINT", stop_print)
hexchat.hook_command("ST_UPDATE", stop_update)
