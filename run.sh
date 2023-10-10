#!bin/bash
kubectl delete -f deployment.yaml -n idaga1
docker-compose build                    
docker-compose push    
kubectl apply -f deployment.yaml -n idaga1            