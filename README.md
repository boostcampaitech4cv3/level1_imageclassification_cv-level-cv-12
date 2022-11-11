# [P stage1] Image Classification Competition

<img width="1085" alt="스크린샷 2022-11-11 오후 1 13 32" src="https://user-images.githubusercontent.com/69153087/201261752-33f6bb6d-ddd0-46fa-a32b-f510c50e5a89.png">


## 일석CV조

**김종해** [T4047] [@Jejugamguel](https://github.com/Jejugamguel) <br/>
**유영준** [T4136] [@youngjun04](https://github.com/youngjun04) <br/>
**이태희** [T4172] [@thlee00](https://github.com/thlee00) <br/>
**조재효** [T4214] [@chojaehyo](https://github.com/chojaehyo) <br/>
**한상준** [T4226] [@jphan32](https://github.com/jphan32) <br/>

<br>

## Project Overview

COVID-19의 확산으로 우리나라는 물론 전 세계 사람들은 경제적, 생산적인 활동에 많은 제약을 가지게 되었습니다. 우리나라는 COVID-19 확산 방지를 위해 사회적 거리 두기를 단계적으로 시행하는 등의 많은 노력을 하고 있습니다. 과거 높은 사망률을 가진 사스(SARS)나 에볼라(Ebola)와는 달리 COVID-19의 치사율은 오히려 비교적 낮은 편에 속합니다. 그럼에도 불구하고, 이렇게 오랜 기간 동안 우리를 괴롭히고 있는 근본적인 이유는 바로 COVID-19의 강력한 전염력 때문입니다.

감염자의 입, 호흡기로부터 나오는 비말, 침 등으로 인해 다른 사람에게 쉽게 전파가 될 수 있기 때문에 감염 확산 방지를 위해 무엇보다 중요한 것은 모든 사람이 마스크로 코와 입을 가려서 혹시 모를 감염자로부터의 전파 경로를 원천 차단하는 것입니다. 이를 위해 공공 장소에 있는 사람들은 반드시 마스크를 착용해야 할 필요가 있으며, 무엇 보다도 코와 입을 완전히 가릴 수 있도록 올바르게 착용하는 것이 중요합니다. 하지만 넓은 공공장소에서 모든 사람들의 올바른 마스크 착용 상태를 검사하기 위해서는 추가적인 인적자원이 필요할 것입니다.

따라서, **우리는 카메라로 비춰진 사람 얼굴 이미지 만으로 이 사람이 마스크를 쓰고 있는지, 쓰지 않았는지, 정확히 쓴 것이 맞는지 자동으로 가려낼 수 있는 시스템이 필요합니다.** 이 시스템이 공공장소 입구에 갖춰져 있다면 적은 인적자원으로도 충분히 검사가 가능할 것입니다.

<br>

## Dataset
- Total : 4500
- 학습 데이터와 평가 데이터전체를 구분하기 위해 임의로 섞어서 분할하였습니다. 
- 학습 데이터셋이 아닌 나머지 40%의 데이터셋 중에서 20%는 public 테스트셋, 그리고 20%는 private 테스트셋으로 사용됩니다.
- Train : 2700
- Test : 1800
- 한 사람당 사진의 개수: 7 [마스크 착용 5장, 이상하게 착용(코스크, 턱스크) 1장, 미착용 1장]
- 이미지 크기: (384, 512)
<br>

## Objective
성별, 연령, 마스크 착용 여부에 따라 사진을 총 18개의 클래스로 분류

<img src="https://user-images.githubusercontent.com/68593821/131881060-c6d16a84-1138-4a28-b273-418ea487548d.png" height="500"/>

<br>

## 🏆 Result
- Rank : 7등 / 20팀
- Private_f1 : 0.7521
- Private_acc : 81.2540

<img width="1050" alt="스크린샷 2022-11-11 오후 1 25 36" src="https://user-images.githubusercontent.com/69153087/201263035-fc44a69f-c35b-4248-8c93-777bfffb1d1f.png">

<br>

## Milestone
<img width="642" alt="스크린샷 2022-11-11 오후 2 12 23" src="https://user-images.githubusercontent.com/69153087/201268107-253b75b5-8cd8-403a-ade9-3262fb67f996.png">

- 프로젝트 수행에 대한 과정과 세부 목표 수립
- Data / Image EDA 을 통해 적합한 모델과 전처리 기법을 고안
- 다양한 Baseline 모델을 학습하여, 결과 비교를 통해 적합한 모델을 선정
- 모델의 인식률을 올릴 수 있는 다양한 학습 기법을 적용
- 최종 결정 모델에 대해 K-Fold를 적용하며, 제출 결과 상위 submission의 Hard Voting 방식의 Ensemble 기법을 적용하여 각 모델 특성의 상호 보완

<br>


## Approach

<img width="700" height="500" alt="screen_shot_2022-11-04_at_11 09 00_am" src="https://user-images.githubusercontent.com/69153087/201267655-98896c0f-12bd-47a4-b6a8-82bc1a15e4e8.png">

## 
