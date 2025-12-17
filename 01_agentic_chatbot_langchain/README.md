
# Work Package 01: App Dev

#########################################
## Task 1: Agentic AI Architecture Design

### Sub-task 01: Primary component selection 

- LLM Model (GPT OpenAI, Claude, or any)
- Embedding model (text-embedding, cohere-multilangual-v3, or any)
- Platform (Langchain, CrewAI, OpenAI Agent, or any)
- Use Agentic Agent or Graph

### Sub-Task 02: App architecture

+ How app data flow organized (multi-agents, number of layers)
+ Define detail each agent and tools support
+ Vector-store for RAG document (vector search method like FAISS) 
+ Block diagram release


#########################################
## Task 2: Coding
### Sub-Task 01: Folder Structure  

 - Code folder structure planning

### Sub-Task 02: Preparation

- Python .venv 
- linux plaform like WSL-ubuntu
- docker installation (linux version/docker desktop)
- Azure account/Resource Group/ACR repo 

### Sub-Task 03: Test primary project components

 - Test OpenAI LLM request
 - Test Embedding model request
 - Test Agent class in coverage of supporting tools such as RAG, computing exchange amount, computing saving amount, connecting DB and get data, etc.
 - Test single question, and conversation 
 - Test docker build image Test ACR repo image upload
 - 
### Sub-Task 04: Integrating form final code

- Integrate separate function dev into app system
- Test call app api


#########################################
## Task 3: Test build docker image + container

### Sub-task 01: Retest requirements.txt --> create python env + run app for test
Note: SHOULD `pip install  -r requirements.txt --no-deps` 
      **--no-deps** : support you passed mismatched version lib when installing.    
Run uvicorn deploy api
Run call api

### Sub-task 02: Test Build docker in local with same platform like in working in ACA
Prepare 3 bash/shell scripts:
1- [.sh] Command to Delete old image with \<none\> in name
2- [.sh] Command to Build docker image with Dockerfile (platform+requirements.txt)
     Bạn có thể gặp lỗi (pylib không tương tích với python version hay platform, thiếu thư viện hay thư viện khác đôi chút với trên win)
3- [.sh] Command to Run image create container (because image stored locally --> not need to pull images)
     Potential Error: 
	     container được built và chạy (run `docker ps -a`  sẽ thấy container với up to N seconds)
	     nhưng bạn gọi API lại ko được, bị lỗi
	 Reason: 
		 uvicorn command in container not run with 0.0.0.0 (accepted all), or port is correct?
     Debug: 
	      Print container log: `docker logs agentic_chatbot_api`
	      Open container terminal to code: 
		      + Run 
			      `docker exec -it agentic_chatbot_api bash` --> Container Terminal >>
		      + Bạn có thể install something in container 
			      `apt update && apt install -y curl`
		      + Sau đó, check curl bên trong container: 
			      `curl http://localhost:8001`
		      + Hay kiểm tra chạy lại lệnh uvicorn bên trong container:
			      `uvicorn app.chatbot:app --host 0.0.0.0 --port 8001`
	         
            Dùng cách trên để debug, tôi xuất log và thấy app tôi đang chạy embedding vector cho tài liệu. Lý do: đường dẫn tới file .faiss có sẵn bị sai. Tôi đã sửa lại đường dẫn, build lại docker image, run container lại và giờ tôi có thể gọi API.\
            
   4- Bạn run 3 shell script.sh trên, khi run xong script thứ 3 deploy container thì bạn chạy
   `docker ps -a` --> show all containers is running and container "up to 1 minutes ..." is running one.
   Để biết chi tiết hơn docker đã copy gì vào container (copy app vào container như nào), bạn mở container terminal ~ bạn vào máy container --> bạn kiểm tra trong máy container nhé!

	     
---
# Work Package 02: Deployment (DevOpt = Repo + CICD Pipelines)

#########################################
## Task 1: Upload App code to Azure Project Repo

#### Sub-task 01 : Code preparation

agentic_chatbot_app
     |-app
        | - requirements.txt
        | - .env
        | - YOUR CODE HERE
     |-docker
         |- Dockerfile
     |-pipelines
         |- azure-pipelines-stage1-infras.yml
         |- azure-pipelines-stage2-docker-image.yml
         |- azure-pipelines-stage3-container.yml

Code upload to repo includes three main things:
- App code (app your deved above + requirements.txt)
- docker (Dockerfile to build image)
- pipelines (here I organize 3 pipelines, you can use just 1 or any depend on)  
	  + CICD1 (pipelines-stage1-infras.yml): creating system resource with terraform 
	  + CICD2 (pipelines-stage2-docker-image.yml): build docker image + up to ACR
	  + CICD3 (pipelines-stage3-container.yml): pull docker image, run container ACA by terraform 

Upload to Repo


#########################################
## Task 2: Create CICD Pipelines

In Project DevOpt winđơ in az, create pipelines and indicate to file azure-pipelines...yml in the repo
For example, you need create 3 pipelines here.
Each pipelines, you can add ENV VAR in UI 

IMPORTANT NOTE:
Pipelines need HW to run:
   1-Microsoft Host pool (Free but 60 mins timeout) 
   2-Selfhost pool (Free - Máy của bạn)
   3-Parralel Microsoft pool (40usd/sku)  
                 
Potential Issues:
Khi chạy pipelines trên host này thì toàn bộ terraform daemon hay docker daemon phải có trên máy đó, kết nối internet đẩy đủ và Az DevOpt có quyền truy cập tới host.

Khi chạy có thể cập nhật ubuntu hoặc download phần mềm, cài package --> có thể bị lỗi lq tới CDN ở các vùng bị chặn bởi VN hay authen request từ VN --> cần đổi url default sang url nào ok.


#########################################
## Task 3: Test pipelines
#### Sub-task 01: Run/Debug/Fix/Rerun (CICD1) azure-pipelines-stage1-infras.yml
Build infras --> DONE

#### Sub-task 02: Run/Debug/Fix/Rerun (CICD2) pipelines-stage2-docker-image.yml
CICD2 --> DONE
+ az login (wsl terminal)
`az login --tenant 42d1f2bd-0d00-4740-b3ae-59320171ec2b --use-device-code`
Then copy link and past provided OTP code --> OK

+ Check image in ACR repository
	 1- Login
	+ az acr login --name acrngothanhnhan125
	 2- Show tags of images
	+ az acr repository show-tags --name acrngothanhnhan125 --repository agentic_chatbot_api --output tsv
	 3- Pull Image to local	
	+ docker pull acrngothanhnhan125.azurecr.io/agentic_chatbot_api:latest
	 4- Get images information (size) 
	+ docker images acrngothanhnhan125.azurecr.io/agentic_chatbot_api

Check image in ACR at local:
`sh run_docker_image_acr.sh` --> run docker image pulled to local machine to test
`python call_chatbot_api.py` --> to check Query API --> DONE

Potential Issue:
- build python venv get mismatch in python version --> add --no-deps
- version not correct with platform (should test in local early)
- authentication when push image to ACR, correct in .yml

#### Sub-task 03: Run/Debug/Fix/Rerun (CICD3) azure-pipelines-stage3-container.yml

Deploy container for chatbot --> DONE

Potential Issues:
- expose port 8000 --> local
- expose to public port 443 --> public. This port default, not need configuring
- Timeout issue relating to ingress/egress


#########################################
## Task 04: Test Container API call

Get container expose API url
Run url API request --> get response --> OK

#########################################

## Task 05: Demo Webapp

streamlit code to call chatbot API 

Test streamlit call API at local first --> deploy to streamlit cloud.

# END



---

### 01. Build docker

sh build_docker_image.sh

### 02. Delete docker image with name <none> reasonned by dupplicated image name.

sh delete_docker_image_name_none.sh

### 03. Run docker image creating container

sh run_docker_image.sh

Script includes remove existed container and rebuilt new container for image.

### Debug code in container

1- check container deployed : docker ps -a  ==> list of container, Up to x minutes... --> still listenning
2- open container terminal ==> check source code
                            |- rerun code in container --> view log



