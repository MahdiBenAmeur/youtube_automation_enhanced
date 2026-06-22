"""Randomly choose and upload either car driving or fighting balls short."""
import random
from src.pipeline.pipeline import build_and_upload_ai_car_driving_short, build_and_upload_short

def upload_random_short():
    choice = random.choice(['car_driving', 'fighting_balls'])
    print(f'Selected: {choice}')

    if choice == 'car_driving':
        print('Uploading AI Car Driving short...')
        result = build_and_upload_ai_car_driving_short()
    else:
        print('Uploading Arena Royale short...')
        result = build_and_upload_short(title='Arena Royale Battle Royale', n_balls=8)

    print(f'Upload result: {result}')
    return result

if __name__ == '__main__':
    upload_random_short()
