# HGBT
## How to run the demo
docker build -t hgpt_server .

docker run -p 5000:5000 -d hgpt_server
## Client example code
python client.py <function name>

upload: show how to call upload api

delete: show how to call delete api

learn: show how to call learn api

chat: show how to call chat api

## API docs
[document](https://openweb3.atlassian.net/wiki/spaces/O/pages/4751361)https://openweb3.atlassian.net/wiki/spaces/O/pages/4751361

## Architect
- app.py
    - main file
    - upload file and store in uploads/
- utils.py
    - tools for other api
    - search
    - weather
    - stock
- demo.py
    - UI base on streamlit