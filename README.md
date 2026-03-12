This is run with docker compose, the tts api relies on the haproxy endpoints and is not configurable

Make sure you have 
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

And you have 
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Then run

`docker compose up -d` in the root of the directory.

This will take ages as the pytorch docker image is huge when built.

to test once it's up
```bash
curl -X GET "http://localhost:5002/tts?identifier=job9001&voice=Example%20Woman&pitch=0" \
     -H "Authorization: vote_goof_2024" \
     -H "Content-Type: application/json"  \
     -d '{"text":"Yeah apc destroyed mission accomplished!"}'\
     --output test.ogg
```

Then play with your choice of player

You can view hte haproxy stats at localhost:8008

The docker compose file is a multi stage build of system deps, python deps and then the actual code

if you want to rebuild python deps for some reason you can with
`docker compose build --build-arg CACHEBUST=$(date +%s)`
