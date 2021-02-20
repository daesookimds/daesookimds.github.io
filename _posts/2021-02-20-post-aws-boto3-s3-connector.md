---
title:  "AWS S3 파일 입출력 자유롭게 이용하기 with Python"
search: true
categories:
  - AWS
  - Python
last_modified_at: 2021-02-20T
---


데이터를 어디에 저장해야하는지 결정하는 것은 데이터를 활용하는 모든 사람들에게 중요한 문제다.
**더욱이 데이터 엔지니어, 사이언티스트들은 데이터를 실제로 수집하고 저장하는 주체로써 데이터의 종류, 수집 방법,
사용 목적, 사용 방식, 사용 주기 등 여러가지 비지니스상 조건에 따라 적절한 도구와 저장소를 선택**할 수 있어야 한다.

이 포스트는 오로지 **데이터 사이언티스트 관점**으로 클라우드 저장공간 중에서 가장 사용이 단순하고 편리하며 그만큼 활용 범위가 넓은 AWS S3(Amazon Simple Storage Service)에 대해 알아보고, Python을 활용하여 파일 입출력 방법을 코드 수준으로 알아본다.

S3는 AWS에서 제공하는 인터넷 스토리지 서비스로써 웹상에 접속할 수 있는 곳이면 어디서든 원하는 양의 데이터를 저장하고 검색할 수 있는 데이터 저장소이다. S3는 기본적으로 버킷, 액세스 포인트(파일이 저장된 경로명)를 통해 데이터 입출력이 이루어진다. 이 이상 S3 서비스에 대한 더 자세한 설명은 아래  링크로 대체하고 바로 코드를 통해 데이터 입출력이 어떻게 손쉽게 가능한지 확인해보도록 하자.


 <https://docs.aws.amazon.com/ko_kr/AmazonS3/latest/dev/Welcome.html>

#### part1: import

첫번째 파트는 패키지 import 부분이다.
python으로 AWS 서비스를 활용하기 위한 boto3
그 외 파일 입출력과 형태변환을 용이하게 하는데 필요한 json, pickle, pandas, io
패키지를 불러온다.

```python

import json
import boto3
import pickle
import pandas as pd
from io import StringIO, BytesIO

```


#### part2: Connect with AWS Access Key

두번째 파트는 AWS 서비스를 사용하기 위한 서비스 연결 부분이다.
AWS 웹상에서 콘솔을 통한 직접적인 서비스 활용이 아닌 외부 라이브러리에서 연결하여 사용하기 위해서는
Access Key가 필요하다. 아래 코드를 보면 연결을 위해 기본적으로 `aws_access_key_id`, `aws_secret_access_key`, `region_name` 세가지가 필요하며 boto3.Session() 으로
서비스 세션을 생성해준다.

**_Access Key는 보안상 매우 엄격하게 관리해야한다. 실제로 Access Key만 추적하여 상시 크롤링을 하고
그렇게 수집/탈취한 다른 사람에 Access Key로 협박을 하거나 남용으로 엄청난 과금 폭탄을 선사하는 해커?가 있다고 한다._**


```python

class TakeAccessKey(object):
    def __init__(self, **param):
        param = {key: value for key, value in param.items()}
        self.aws_access_key_id = param['aws_access_key_id']
        self.aws_secret_access_key = param['aws_secret_access_key']
        self.region_name = param['region_name']
        self.session = boto3.Session(aws_access_key_id=self.aws_access_key_id,
                                     aws_secret_access_key=self.aws_secret_access_key, region_name=self.region_name)

```

#### part3: S3 활용

S3Connector 클래스는 part2 에서 정의한 TakeAccessKey 클래스를 상속받아 작성되었다.
연결된 boto3 세션으로 s3.client, s3.resource를 불러올 수 있으며,
이 client와 resource를 활용하여 download, upload를 실행할 수 있다.

download, upload 모두 버킷(bucket_nm), 액세스 포인트(remote_fn)를 입력하면
자동으로 파일확장자(아래 클래스는 json, csv, xlsx, pkl 지원)를 확인하여 형태에 맞게
저장 및 다운로드를 실행할 수 있다.


```python
class S3Connector(TakeAccessKey):
    def __init__(self, **param):
        super().__init__(**param)
        self.s3_client = self.session.client('s3')
        self.s3_resource = self.session.resource('s3')


    def get_bucket_list(self):
        self.buckets = [ins['Name'] for ins in self.s3_client.list_buckets()['Buckets']]

        return self.buckets


    def download(self, bucket_nm, remote_fn):
        '''
        download file from AWS S3
        :param bucket_nm: S3 bucket name you want to download
        :param remote_fn: S3 file path you want to download
        :return: download data
        '''
        file_nm = remote_fn.split('/')[-1]
        file_type = file_nm.split('.')[-1]
        content_obj = self.s3_resource.Object(bucket_nm, remote_fn).get()

        if file_type == 'json':
            content = content_obj['Body'].read().decode('utf-8')
            data = json.loads(content)
        elif file_type == 'csv':
            content = content_obj['Body'].read().decode('utf-8')
            data = pd.read_csv(StringIO(content))
        elif file_type == 'xlsx':
            content = content_obj['Body'].read()
            data = pd.read_excel(content)
        elif file_type == 'pkl':
            content = content_obj['Body'].read()
            data = pickle.loads(content)
        else:
            content = content_obj['Body'].read().decode('utf-8')
            data = content

        return data


    def upload(self, bucket_nm, remote_fn, upload_data):
        '''
        upload file to AWS S3
        :param data: A data you want to upload
        :param bucket_nm: S3 bucket name you want to upload
        :param remote_fn: S3 file path you want to upload
        :param type: Type of data you want to upload
        :return: None
        '''
        type = remote_fn.split('.')[-1]
        bucket = self.s3_resource.Bucket(bucket_nm)

        if type == 'json':
            bucket.Object(key=remote_fn).put(Body=json.dumps(upload_data))
        elif type == 'csv':
            buffer = StringIO()
            upload_data.to_csv(buffer, index=False)
            bucket.Object(key=remote_fn).put(Body=buffer.getvalue())
        elif type == 'xlsx':
            buffer = BytesIO()
            upload_data.to_excel(buffer, index=False)
            bucket.Object(key=remote_fn).put(Body=buffer.getvalue())
        elif type == 'pkl':
            bucket.Object(key=remote_fn).put(Body=pickle.dumps(upload_data))
        else:
            buffer = StringIO()
            buffer.write(upload_data)
            bucket.Object(key=remote_fn).put(Body=buffer.getvalue())

```


<br>
Python으로 S3를 활용하는 작업은 AWS를 클라우드 서비스로 채택하는 경우 비일비재하다.
그때 마다 코드를 재작성하지 않고 패키지로 불러와 사용하기 위해 작성되었으며,
기본적인 내용이지만 AWS를 처음 접하는 누군가에게는 도움이 되었으면 한다.
