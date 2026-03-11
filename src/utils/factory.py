'''
파일명칭 : factory.py
기    능 : 기존의 구현에서는 하나의 특정 모델(MoviePredictor)을 위한 코드로 새로운 모델이 추가되거나 
           다른 모델로 변경해서 실험을 진행하려면 많은 곳에서 코드 수정이 일어나야 함. 
           모델 뿐만이 아니라 다른 설정들 또한 동일한 문제를 해결하기 위해서
           Factory 패턴과 Dynamic Import를 결합하여 인자로 받은 모델명을 검증하고, 
           해당 클래스를 동적으로 가져와 반환하도록 하여 확장성을 높일 수 있음.
입    력 : 
출    력 : 
작 성 자 : 송 용 단
작성일자 : 2026-03-11
'''
from src.model.movie_predictor import MoviePredictor

class ModelFactory:
    _models = {
        "movie_predictor": MoviePredictor,
    }

    @classmethod
    def validate_and_get(cls, model_name: str):
        """이름을 검증하고 해당 모델 클래스를 반환"""
        name_lower = model_name.lower()
        if name_lower not in cls._models:
            valid_options = list(cls._models.keys())
            raise ValueError(
                f"Invalid model name: '{model_name}'. "
                f"Available: {valid_options}"
            )
        return cls._models[name_lower]

    @classmethod
    def create(cls, model_name: str, **kwargs):
        model_class = cls.validate_and_get(model_name)
        return model_class(**kwargs)
