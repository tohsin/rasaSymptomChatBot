# HI Meet LOUISE👨🏾‍🎤

## Description
Louise is a medical chat bot made to run or be deployed on telegram or just a webserver,
based on other medical chat bots where a lot of intents example are wriiten to map to specific 
symptoms or diagnosis , Louise takes a diffrent approach.

Louise is more adaptive because there are seperate predictive models ones for prediting the 
diagnosis from the symptoms and the other to interact with the user and extract by context what 
a user might be describing as a symptom, this allows for scalability of the chat bot as if a new diseases is added one doesnt need to write new intents to describe this.

The language processing and chat bot runs a transformer newtwork and the classification uses a SVC for classification.


# setup
TO install rasa follow guide from
```
https://rasa.com/docs/rasa/installation/
```

