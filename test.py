from main import (
    AbstractFirstStage,
    Minimize
)


class TestFirstStage(AbstractFirstStage):
    """
    Тестирование на данных которые есть в файле курсовой от Полякова
    Также это референс для тех кто не знает как генерировать данные.
    Те кто не занют как генерировать данные, то соболезную вам нужно вручную составить таблицу истинности
    т.к составляете все в ручную то достаточно использовать в методе столбец с вариантами логических перменных
    и столбец со значениями функции. Понятное дело важно, чтобы первое значение в одном списке было
    связанно с первым во втором, второе значение со вторым значением и так далее
    """
    def make_table(self):
        """
        Юзаются только 2 столбца с вариантами и
        :return:
        """
        column1 = [
            '00000',
            '00001',
            '00010',
            '00011',
            '00100',
            '00101',
            '00110',
            '00111',
            '01000',
            '01001',
            '01010',
            '01011',
            '01100',
            '01101',
            '01110',
            '01111',
            '10000',
            '10001',
            '10010',
            '10011',
            '10100',
            '10101',
            '10110',
            '10111',
            '11000',
            '11001',
            '11010',
            '11011',
            '11100',
            '11101',
            '11110',
            '11111',
        ]
        column2 = [
            'd',
            '1',
            '1',
            '0',
            'd',
            '0',
            '1',
            '1',
            'd',
            '0',
            '1',
            '1',
            'd',
            '0',
            '0',
            '1',
            'd',
            '1',
            '1',
            '0',
            'd',
            '1',
            '1',
            '0',
            'd',
            '0',
            '1',
            '1',
            'd',
            '0',
            '1',
            '1'
        ]
        # опять же не забудь преобразовать данные во фрейм
        table = AbstractFirstStage.make_df_from_table(self.indexes, [column1, column2])
        return table


def test():
    """
    Пример со значениями Полякова
    :return:
    """
    first_stage = TestFirstStage(
        ['x1x2x3x4x5', 'f'], 5)
    first_stage.write('truth_table_test.csv')
    print('КДНФ:', first_stage.kdnf_str)
    print('ККНФ:', first_stage.kknf_str)

    minimize = Minimize(first_stage)
    minimize.write(minimize.cube_table_implicant, 'cube_test.csv')
    minimize.write(minimize.table_essential_implicants, 'essential_implicants_test.csv')


if __name__ == '__main__':
    test()
