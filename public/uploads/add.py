import os as os

def combineFeaFile(f_path, out):
    dict = {}
    label = []
    f_paths = getAllFeaPath(f_path)
    for path in f_paths:
        fea_name = path.split("/")[-1]
        fea_name = fea_name[0:len(fea_name) - 4]
        r = open(path, mode='r', encoding='utf-8')
        lines = r.readlines()
        file_ctx = []
        if label == []:
            for line in lines:
                line = line.strip()
                ctxs = line.split(",")
                if len(ctxs) == 1:
                    ctxs = line.split(" ")
                _label = ctxs[0]
                label.append(_label)
                line_ctx = ctxs[1:len(line)]
                file_ctx.append(line_ctx)
            dict[fea_name] = file_ctx
        else:
            for line in lines:
                line = line.strip()
                ctxs = line.split(",")
                if len(ctxs) == 1:
                    ctxs = line.split(" ")
                line_ctx = ctxs[1:len(line)]
                file_ctx.append(line_ctx)
            dict[fea_name] = file_ctx
    w = open(out, mode='w', encoding='utf-8')
    fea_names = dict.keys()
    for i in range(0, len(label)):
        ctx = ''
        ctx += label[i]
        for fea_name in fea_names:
            line_ctxs = dict[fea_name][i]
            for line_ctx in line_ctxs:
                ctx += ',' + line_ctx
        w.write(ctx.strip() + '\n')
    w.close()
    return out

def getAllFeaPath(path):
    pathList = []
    for filename in os.listdir(path):
        pathList.append(os.path.join(path, filename))
    return pathList


combineFeaFile("E:/深度学习/graduationDesign/data/Feature/Feature_data/add", "E:/深度学习/graduationDesign/data/Feature/Feature_data/conbine.csv");