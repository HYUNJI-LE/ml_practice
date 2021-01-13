import struct #변환프로그램

def to_csv(name, maxdata):
    #레이블 파일과 이미지 열기
    lbl_f = open("./mnist/"+name+"-labels-idx1-ubyte","rb") #이미지 숫자 정답 파일 #rb : 바이너리 파일 읽기
    img_f = open("./mnist/"+name+"-images-idx3-ubyte", "rb") #이미지 파일
    csv_f = open("./mnist/"+name+".csv", "w", encoding="utf-8")
    #헤더 정보 읽기
    mag, lbl_count = struct.unpack(">II", lbl_f.read(8))
    mag, img_count = struct.unpack(">II", img_f.read(8))
    rows, cols = struct.unpack(">II", img_f.read(8))
    pixels = rows*cols
    #이미지 데이터를 읽고 CSV로 저장하기(2개의 파일에서 하나씩 이미지와 숫자를 꺼냄)
    for idx in range(lbl_count):
        if idx > maxdata: break
        label = struct.unpack("B", lbl_f.read(1))[0] #unpack 정수로 변환
        bdata = img_f.read(pixels)
        sdata = list(map(lambda n: str(n),bdata))
        csv_f.write(str(label)+",")
        csv_f.write(",".join(sdata)+"\r\n")
        #잘 저장되었는지 이미지 파일로 저장해서 테스트하기
        if idx < 10:
            s= "P2 28 28 255/n"
            s+= " ".join(sdata)
            iname = "./mnist/{0}-{1}-{2}.pgm".format(name,idx,label)
            with open(iname, "w", encoding="utf-8") as f:
                f.write(s)
    csv_f.close()
    lbl_f.close()
    img_f.close()
to_csv("train",1000) #일단 부분만 저장
to_csv("t10k",500)