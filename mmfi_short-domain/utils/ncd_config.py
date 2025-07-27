def get_mmfi_ncd_config():
    """
    MMFI-27 NCD配置
    每个任务定义包含的类别和对应的人员
    """
    ncd_config = {
        # Task 0: 15个类，只有Person 1
        0: {
            'new_classes': list(range(15)),
            'data': {i: [1] for i in range(15)}
        },
        
        # Task 1: 3个新类(15-17) + 旧类(0-14)增加Person 2
        1: {
            'new_classes': [15, 16, 17],
            'data': {
                **{i: [1, 2] for i in range(15)},  # 旧类增加Person 2
                **{i: [1] for i in range(15, 18)}   # 新类只有Person 1
            }
        },
        
        # Task 2: 3个新类(18-20) + 15-17增加Person 2
        2: {
            'new_classes': [18, 19, 20],
            'data': {
                **{i: [1, 2] for i in range(18)},   # 0-17都有Person 1,2
                **{i: [1, 2] for i in range(18, 21)} # 新类有Person 1,2
            }
        },
        
        # Task 3: 3个新类(21-23) + 所有旧类增加Person 3
        3: {
            'new_classes': [21, 22, 23],
            'data': {
                **{i: [1, 2, 3] for i in range(21)},  # 旧类增加Person 3
                **{i: [1, 2] for i in range(21, 24)}   # 新类有Person 1,2
            }
        },
        
        # Task 4: 3个新类(24-26) + 21-23增加Person 3
        4: {
            'new_classes': [24, 25, 26],
            'data': {
                **{i: [1, 2, 3] for i in range(24)},     # 0-23都有Person 1,2,3
                **{i: [1, 2, 3] for i in range(24, 27)}  # 新类有Person 1,2,3
            }
        },
        
        # Task 5: 无新类，所有类增加Person 4
        5: {
            'new_classes': [],
            'data': {i: [1, 2, 3, 4] for i in range(27)}
        },
        
        # Task 6: 无新类，0-13增加Person 5
        6: {
            'new_classes': [],
            'data': {
                **{i: [1, 2, 3, 4, 5] for i in range(14)},
                **{i: [1, 2, 3, 4] for i in range(14, 27)}
            }
        },
        
        # Task 7: 无新类，14-26增加Person 5
        7: {
            'new_classes': [],
            'data': {i: [1, 2, 3, 4, 5] for i in range(27)}
        },
        
        # Task 8: 巩固任务
        8: {
            'new_classes': [],
            'data': {i: [1, 2, 3, 4, 5] for i in range(27)}
        }
    }
    
    return ncd_config