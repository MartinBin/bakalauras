# bakalauras

To run frontend you need Node and npm.
Then go to Font\Frontend folder

Run:
```
npm install
npm run dev
```
 
________________________________________________________
To install Pytorch with Cuda support:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Then run in root folder:
```
pip3 install -r ./requirements.txt
```

It will install remaining packages needed for python.

Then go to server folder and run:
```
python ./manage.py makemigrations
python ./manage.py migrate
python ./manage.py runserver
```
______________________________________________________

Also Docker is needed for mongo databas

Go to root folder and run:
```
docker compose up -d
```