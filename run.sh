#!bin/bash
kubectl delete -f manifest.yaml -n idaga1
docker-compose build                    
docker-compose push    
kubectl apply -f manifest.yaml -n idaga1            