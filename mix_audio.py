import os
import numpy as np
import librosa
import soundfile as sf
import re
from glob import glob
import math

# 폴더 경로 설정
target_dir = 'target_audio'
noise_dir = 'noise_audio'
output_dir = 'mixed_audio'

# 출력 폴더가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# 파일 경로 가져오기
target_files = sorted(glob(os.path.join(target_dir, '*.wav')))
noise_files = sorted(glob(os.path.join(noise_dir, '*.wav')))

# SNR 레벨 설정 (단위: dB)
snr_levels = {
    'easy': 6,     # 조용한 사무실 환경 (음성이 소음보다 훨씬 큼)
    'normal': 2,   # 일반적인 생활소음 환경 (음성이 소음보다 약간 큼)
    'hard': -1       # 소음이 큰 환경 (음성과 소음이 같은 크기)
}

# 소리의 크기(RMS power) 계산 함수
def calculate_rms(audio):
    """오디오 신호의 RMS(Root Mean Square) 값을 계산합니다."""
    return np.sqrt(np.mean(np.square(audio)))

# 원하는 SNR로 소음 조절 함수
def adjust_noise_to_snr(speech, noise, target_snr):
    """목표 SNR에 맞게 소음 레벨을 조정합니다."""
    speech_rms = calculate_rms(speech)
    noise_rms = calculate_rms(noise)
    
    # SNR = 10 * log10(speech_power / noise_power)
    # noise_power = speech_power / (10^(SNR/10))
    target_noise_rms = speech_rms / (10 ** (target_snr / 10))
    
    # 조정 계수 계산
    adjustment_factor = target_noise_rms / noise_rms
    
    # 소음 조정
    adjusted_noise = noise * adjustment_factor
    
    return adjusted_noise

# 파일명에서 번호 추출 함수
def extract_number(filename):
    """파일명에서 숫자 부분을 추출합니다."""
    match = re.search(r'(\d+)', os.path.basename(filename))
    if match:
        return match.group(1)
    else:
        return None

# 각 음성 파일에 대해 처리
for i, target_file in enumerate(target_files):
    # 음성 파일 로드
    speech, sr = librosa.load(target_file, sr=None)
    print(f"처리 중: {target_file}")
    
    # 소음 파일 선택 (순환적으로 사용)
    noise_file = noise_files[i % len(noise_files)]
    
    # 소음 파일 로드
    noise, noise_sr = librosa.load(noise_file, sr=None)
    
    # 샘플링 레이트가 다르면 소음 파일 리샘플링
    if sr != noise_sr:
        noise = librosa.resample(noise, orig_sr=noise_sr, target_sr=sr)
    
    # 음성 길이에 맞추어 소음 조정 (반복 또는 자르기)
    if len(noise) < len(speech):
        # 소음이 더 짧으면 반복
        repetitions = math.ceil(len(speech) / len(noise))
        noise = np.tile(noise, repetitions)
    
    # 음성 길이에 맞게 소음 자르기
    noise = noise[:len(speech)]
    
    # 파일 번호 추출
    file_number = extract_number(target_file)
    if file_number is None:
        file_number = f"{i+1:04d}"  # 숫자를 찾을 수 없으면 인덱스 사용
    else:
        # 숫자를 찾았으면 4자리로 형식화
        file_number = f"{int(file_number):04d}"
    
    # 각 SNR 레벨에 대해 합성
    for level_name, snr_value in snr_levels.items():
        # 소음 조정
        adjusted_noise = adjust_noise_to_snr(speech, noise, snr_value)
        
        # 음성과 소음 합성
        mixed_audio = speech + adjusted_noise
        
        # 클리핑 방지를 위한 정규화
        max_abs_value = np.max(np.abs(mixed_audio))
        if max_abs_value > 1.0:
            mixed_audio = mixed_audio / max_abs_value
        
        # 출력 파일 이름
        output_filename = f"{level_name}_{file_number}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # 파일 저장
        sf.write(output_path, mixed_audio, sr)
        
        print(f"생성됨: {output_path}")

print("모든 파일 처리 완료!")