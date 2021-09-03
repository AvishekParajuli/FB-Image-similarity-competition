BATCH_SIZE =16
IM_SIZE = (160,160)
imsize=IM_SIZE[0]
LR =0.001
ALPHA = 0.8
DEBUG=False

anchor_file_list = './list_files/subset_1_queries'
ref_file_list = './list_files/subset_1_references'
#ref_img_dir = "D:/prjs/im-similarity/data/reference"
#anchor_img_dir = "D:/prjs/im-similarity/data/query"
ref_img_dir = "C:/Users/parajav/PycharmProjects/isc/reference/reference"
anchor_img_dir = 'C:/Users/parajav/PycharmProjects/isc/query'



'''
    aa('--anchor_file_list', default='./list_files/subset_1_queries', help="CSV file with query image filenames")
    aa('--anchor_img_dir', default="D:/prjs/im-similarity/data/query", help="search image files in this directory")
    aa('--ref_file_list', default='./list_files/subset_1_references', help="CSV file with reference imagenames")
    aa('--ref_img_dir', default="D:/prjs/im-similarity/data/reference", help="search image files in this directory")
    '''

def getImIds(file_list, image_dir, i0=0, i1=-1):
    image_ids = [l.strip() for l in open(file_list, "r")]

    if i1 == -1:
        i1 = len(image_ids)
    image_ids = image_ids[i0:i1]

    # full path name for the image
    image_dir = image_dir
    if not image_dir.endswith('/'):
        image_dir += "/"

    # add jpg suffix if there is none
    image_list = [
        image_dir + fname if "." in fname else image_dir + fname + ".jpg"
        for fname in image_ids
    ]

    print(f"  found {len(image_list)} images")
    return image_list,image_ids

q_image_list,q_ids = getImIds(anchor_file_list,anchor_img_dir)
ref_image_list, ref_ids = getImIds(ref_file_list, ref_img_dir)

Q_List = q_image_list
R_List = ref_image_list
Q_IDS = q_ids
REF_IDS = ref_ids