import pdb


def get_character_names(args):
    if args.is_train:
        """
        Put the name of subdirectory in retargeting/datasets/Mixamo as [[names of group A], [names of group B]]
        """
        # characters = [['Aj', 'BigVegas', 'Kaya', 'SportyGranny'],
        #               ['Malcolm_m', 'Remy_m', 'Maria_m', 'Jasper_m', 'Knight_m',
        #                'Liam_m', 'ParasiteLStarkie_m', 'Pearl_m', 'Michelle_m', 'LolaB_m',
        #                'Pumpkinhulk_m', 'Ortiz_m', 'Paladin_m', 'James_m', 'Joe_m',
        #                'Olivia_m', 'Yaku_m', 'Timmy_m', 'Racer_m', 'Abe_m']]

        # characters = [['ch01'], ['ch02']]
        # characters = [['Talking_With_Hands'], ['Trinity']]
        characters = [['Trinity'], ['ZEGGS']]
        # characters = [['Talking_With_Hands'], ['ZEGGS']]

    else:
        """
        To run evaluation successfully, number of characters in both groups must be the same. Repeat is okay.
        """
        # characters = [['BigVegas', 'BigVegas', 'BigVegas', 'BigVegas'],  ['Mousey_m', 'Goblin_m', 'Mremireh_m', 'Vampire_m']]
        # tmp = characters[1][args.eval_seq]
        # characters[1][args.eval_seq] = characters[1][0]
        # characters[1][0] = tmp

        pass    # 20230320

    return characters


def create_dataset(args, character_names=None):
    from datasets.combined_motion import TestData, MixedData

    if args.is_train:       # 1
        return MixedData(args, character_names)
    else:
        return TestData(args, character_names)


def get_test_set():
    with open('./datasets/Mixamo/test_list.txt', 'r') as file:
        list = file.readlines()
        list = [f[:-1] for f in list]
        return list


# def get_train_list():
#     with open('./datasets/Mixamo/train_list.txt', 'r') as file:
#         list = file.readlines()
#         list = [f[:-1] for f in list]
#         return list


if __name__ == '__main__':
    '''
    python ./datasets/__init__.py
    '''
    import sys
    [sys.path.append(i) for i in ['./', '../', "../../utils"]]
    import option_parser
    args = option_parser.get_args()
    characters = get_character_names(args)      # [['ch01'], ['ch02']]
    # dataset = create_dataset(args, characters)
    # from torch.utils.data.dataloader import DataLoader
    # data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # for step, motions in enumerate(data_loader):
    #     pass
        # print(motions.shape)        # [[[55, 91, 64], [55]], [[55, 91, 64], [55]]]

    # Test
    from datasets.combined_motion import TestData
    dataset = TestData(args, characters)
    from torch.utils.data.dataloader import DataLoader

    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # pdb.set_trace()
    for step, motions in enumerate(data_loader):
        # pdb.set_trace()
        print(motions.shape)
