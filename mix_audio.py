import os
import numpy as np
import librosa
import soundfile as sf
import re
from glob import glob
import math
import multiprocessing
from tqdm import tqdm

# 폴더 경로 설정
target_dir = 'SER_IS2025'
noise_dir = 'noise_audio'

# SNR 레벨 설정 (단위: dB)
snr_levels = {
    'easy': 6,     # 조용한 사무실 환경 (음성이 소음보다 훨씬 큼)
    'normal': 2,   # 일반적인 생활소음 환경 (음성이 소음보다 약간 큼)
    'hard': -1     # 소음이 큰 환경 (음성과 소음이 같은 크기)
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

# 파일명에서 .wav 확장자만 제거하는 함수
def get_filename_without_extension(filepath):
    """파일 경로에서 확장자를 제외한 파일명만 반환합니다."""
    base_name = os.path.basename(filepath)
    name_without_ext = os.path.splitext(base_name)[0]
    return name_without_ext

# 단일 작업 처리 함수 (병렬 처리용)
def process_combination(args):
    target_file, noise_file, level_name, snr_value, idx = args
    
    try:
        # 음성 파일 로드
        speech, sr = librosa.load(target_file, sr=None)
        
        # 노이즈 파일 로드
        noise, noise_sr = librosa.load(noise_file, sr=None)
        
        # 노이즈 이름 추출 (.wav 제거)
        noise_id = get_filename_without_extension(noise_file)
        
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
        
        # 소음 조정
        adjusted_noise = adjust_noise_to_snr(speech, noise, snr_value)
        
        # 음성과 소음 합성
        mixed_audio = speech + adjusted_noise
        
        # 클리핑 방지를 위한 정규화
        max_abs_value = np.max(np.abs(mixed_audio))
        if max_abs_value > 1.0:
            mixed_audio = mixed_audio / max_abs_value
        
        # 타겟 파일명 추출
        target_basename = get_filename_without_extension(target_file)
        
        # 출력 폴더명 설정 ({난이도}_{노이즈명} 형식)
        output_dir = f"{level_name}_{noise_id}"
        
        # 출력 폴더가 없으면 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 출력 파일 이름 (원본 파일명 유지)
        output_filename = f"{target_basename}.wav"
        output_path = os.path.join(output_dir, output_filename)
        
        # 파일 저장
        sf.write(output_path, mixed_audio, sr)
        
        return output_path
    except Exception as e:
        return f"Error processing {target_file} with {noise_file} at {level_name} level: {str(e)}"

def main():
    # 파일 경로 가져오기
    target_files = sorted(glob(os.path.join(target_dir, '*.wav')))
    noise_files = sorted(glob(os.path.join(noise_dir, '*.wav')))
    
    print(f"총 {len(target_files)}개의 타겟 오디오, {len(noise_files)}개의 노이즈 파일, {len(snr_levels)}개의 SNR 레벨")
    
    # 모든 작업 조합 생성
    tasks = []
    for i, target_file in enumerate(target_files):
        for j, noise_file in enumerate(noise_files):
            for level_name, snr_value in snr_levels.items():
                task_idx = len(tasks) + 1
                tasks.append((target_file, noise_file, level_name, snr_value, task_idx))
    
    total_combinations = len(tasks)
    print(f"총 {total_combinations}개의 조합을 처리할 예정입니다.")
    
    # 병렬 처리를 위한 CPU 코어 수 설정 (사용 가능한 코어의 80%를 사용)
    num_processes = max(1, int(multiprocessing.cpu_count() * 0.8))
    print(f"병렬 처리에 {num_processes}개의 프로세스를 사용합니다.")
    
    # 멀티프로세싱 실행
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_combination, tasks), total=total_combinations, desc="오디오 합성 중"))
    
    # 성공 및 실패 결과 확인
    success_count = sum(1 for r in results if not r.startswith("Error"))
    error_count = sum(1 for r in results if r.startswith("Error"))
    
    print("\n모든 파일 처리 완료!")
    print(f"총 {success_count}개의 합성 오디오 파일이 생성되었습니다.")
    
    if error_count > 0:
        print(f"처리 중 {error_count}개의 오류가 발생했습니다.")
        # 에러 출력 (처음 10개만)
        errors = [r for r in results if r.startswith("Error")]
        for i, error in enumerate(errors[:10]):
            print(f"  오류 {i+1}: {error}")
        if len(errors) > 10:
            print(f"  그 외 {len(errors) - 10}개의 오류가 더 있습니다.")

if __name__ == "__main__":
    main()