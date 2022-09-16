import re
import copy
import pandas as pd

# 한 문장에 대해 input & label 생성하는 함수
def preprocess(sent):
    token_list = sent.split(' ')  # 문장을 공백 단위로 토큰화
    temp_sent = ' '.join(token_list).strip()  # 문장의 양 끝에 있는 공백 제거
    word_pos_label_list = []  # 문장에 대한 label

    # 공백 단위로 토큰화 된 문장에서 토큰을 하나씩 살펴보며 단어와 Pos 태그의 매핑을 찾는다.
    for i in range(len(token_list)):

        token = token_list[i]
        #         print('token', token)
        label_list = None

        # 단어가 단일인 경우: ex> <한지원:PS>
        if '<' in token and '>' in token:
            label_list = [token]

            # 단어에 대한 Pos가 있는 경우: ex> <한지원:PS>
            if ':' in token:
                POS = token.split('>')[0].split(':')[-1]  # 영단어로 이루어진 Pos 태그 파악: ex> PS
                if POS != '':
                    word = ' '.join(label_list).split('<')[-1].split(':')[0]  # 단어 파악: ex> 한지원
                    word_pos_label_list.append(word + '[' + POS + ']')  # label 생성: ex> 한지원[PS]
                else:
                    pass
            # <> 표시만 있고 Pos가 없는 경우: ex> <서울>
            # 이 경우는 단어와 Pos 태그가 매핑되지 않았기 때문에 pass
            else:
                pass

        # 여러 개의 단어로 이루어진 경우: ex> <열 두명 미국인:QT>
        elif '<' in token:
            # 해당 토큰에서 token_list의 끝 토큰까지를 범위로 하여
            # '<'로 시작하는 토큰의 인덱스 찾기: ex> '<열' 토큰의 인덱스 찾기
            first_idx = token_list[i:].index(token)

            # 해당 토큰에서 token_list의 끝 토큰까지를 범위로 하여
            # '>'로 끝나는 토큰의 인덱스 찾기: ex> '미국인:QT>' 토큰의 인덱스 찾기
            for tok in token_list[i:]:
                # '>'로 끝나는 토큰의 인덱스 찾음
                if '>' in tok:
                    last_idx = token_list[i:].index(tok)
                    break
                # 2448번째 문장에 대한 예외 처리 => 내가본 가장 재미있게본 애니메이션<1위:QT>> <!
                else:
                    last_idx = first_idx

            label_list = token_list[i:][first_idx:last_idx + 1]  # 해당 인덱스의 단어들 가져오기: ex> ['<열', '두명', 미국인:QT>']
            n = len(label_list)  # 단어의 길이 측정: ex> n=2

            # 단어에 대한 Pos가 있는 경우: ex> <열 두명 미국인:QT>
            if ':' in label_list[last_idx]:
                POS = label_list[last_idx].split('>')[0].split(':')[-1]  # 영단어로 이루어진 Pos 태그 파악: ex> QT
                word = ' '.join(label_list).split('<')[-1].split(':')[0]  # 단어 파악: ex> 열 두명 미국인
                word_pos_label_list.append(word + '[' + POS + ']')  # label 생성: ex> 열 두명 미국인[QT]

            # <> 표시만 있고 Pos가 없는 경우: ex> <스 타 벅스>
            # 이 경우는 단어와 Pos 태그가 매핑되지 않았기 때문에 pass
            else:
                pass

    label_ch_sent = '\t'.join(word_pos_label_list)  # 문장 sent에 대한 각 label을 '\t'으로 join
    return ' '.join(token_list), label_ch_sent


# input & label로 구성된 dataframe 생성하는 함수
def get_dataframe(lines):
    input_text_list = []
    label_text_list = []

    for i, sent in enumerate(lines):
        input_text, label_text = preprocess(sent)
        input_text_list.append(input_text)
        label_text_list.append(label_text)

    my_dict = {'input': input_text_list, 'label': label_text_list}

    df = pd.DataFrame(my_dict)

    return df


# 영단어로 이루어진 Pos 태그를 특수 기호로 매핑하는 함수
def pos_tag(x, mapping_pos):
    for i in range(len(x)):
        temp = x[i]

        for pos in ['[PS]', '[TI]', '[LC]', '[OG]', '[QT]', '[DT]']:
            if pos in temp:
                temp = temp.replace(pos, mapping_pos[pos])

        x[i] = temp

    return ' '.join(x)


# 모델 학습을 위한 데이터 셋 생성하는 함수
def get_complete_dataframe(df, file_path):
    df['label'] = df['label'].apply(lambda x: pos_tag(x.split('\t'), mapping_pos))
    df.to_csv(file_path)
    return df

if __name__=="__main__":
    file_name = ''
    file_path = ''

    f = open(file_name, 'r', encoding='UTF8')
    lines = f.readlines()

    # 영단어로 이루어진 Pos 태그를 특수 기호로 매핑하기 위한 dictionary
    mapping_pos = {'[PS]': '@', '[TI]': '#', '[LC]': '$', \
                   '[OG]': '&', '[QT]': '`', '[DT]': '='}

    temp_df = get_dataframe(lines)

    df = get_complete_dataframe(temp_df, file_path)