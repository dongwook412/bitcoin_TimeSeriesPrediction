# 일별 비트코인 데이터를 이용한 시계열 예측
- 2018년부터 2020년까지의 일별 비트코인 가격 데이터를 통해 미래 시점의 가격을 예측하고자 함
- 통계적 시계열 모형인 ARIMA에 Seasonality를 고려한 SARIMA를 사용하여 Regression 수행
- 잔차에 대한 ACF/PACF를 확인하여 자기 상관을 조사
- 훈련 데이터를 적합한 모델에 테스트 데이터를 대입하여 예측 및 RMSE 비교
- 변동성에 대한 예측을 수행하는 ARCH와 GARCH를 사용하여 Volatility Clustering 현상 확인
<p align="center"><img src="./img/1.PNG"></p>
<br/>


**[뱃지나 프로젝트에 관한 이미지들이 이 위치에 들어가면 좋습니다]**  
One Paragraph of project description goes here / 프로젝트의 전반적인 내용에 대한 요약을 여기에 적습니다

## Getting Started / 어떻게 시작하나요?

이 곳에서 설치에 관련된 이야기를 해주시면 좋습니다.

### Prerequisites / 선행 조건

아래 사항들이 설치가 되어있어야합니다.

```
예시
```

### Installing / 설치

아래 사항들로 현 프로젝트에 관한 모듈들을 설치할 수 있습니다.

```
예시
```

## Running the tests / 테스트의 실행

어떻게 테스트가 이 시스템에서 돌아가는지에 대한 설명을 합니다

### 테스트는 이런 식으로 동작합니다

왜 이렇게 동작하는지, 설명합니다

```
예시
```

### 테스트는 이런 식으로 작성하시면 됩니다

```
예시
```

## Deployment / 배포

Add additional notes about how to deploy this on a live system / 라이브 시스템을 배포하는 방법

## Built With / 누구랑 만들었나요?

* [이름](링크) - 무엇 무엇을 했어요
* [Name](Link) - Create README.md

## Contributiong / 기여

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us. / [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) 를 읽고 이에 맞추어 pull request 를 해주세요.

## License / 라이센스

This project is licensed under the MIT License - see the [LICENSE.md](https://gist.github.com/PurpleBooth/LICENSE.md) file for details / 이 프로젝트는 MIT 라이센스로 라이센스가 부여되어 있습니다. 자세한 내용은 LICENSE.md 파일을 참고하세요.

## Acknowledgments / 감사의 말

* Hat tip to anyone whose code was used / 코드를 사용한 모든 사용자들에게 팁
* Inspiration / 영감
* etc / 기타
