def open_txt_to_list(root_dir):
    data = []
    file = open(root_dir, 'r')
    file_data = file.readlines()
    for element in file_data:
        temp1 = element.strip('\n')
        temp2 = temp1.strip('\t')
        data.append(temp2)
    int_data = list(map(int, data))
    return int_data

def get_timestamped_simplices(root_nverts, root_timestamp, root_simplices):
    data_nverts = open_txt_to_list(root_nverts)
    data_timestamp = open_txt_to_list(root_timestamp)
    data_simplices = open_txt_to_list(root_simplices)
    data_simplices_split = []
    for i in range(len(data_nverts)):
        temp = data_simplices[0: data_nverts[i]]
        temp.sort()
        data_simplices_split.append(temp)
        del data_simplices[0: data_nverts[i]]
        if len(data_simplices) == 0:
            print('二者是对应的哦！')
    return data_timestamp, data_simplices_split














