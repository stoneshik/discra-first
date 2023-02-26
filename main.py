from abc import ABC, abstractmethod
import itertools

import pandas as pd


class AbstractFirstStage(ABC):
    """
    Суперкласс для классов генерящих данные таблицы истинности.
    Для генерации данных своего варианта отнаследуйся от этого класса и
    переопредели метод make_table
    """
    def __init__(self, indexes: list, num_logic_var: int):
        """
        :param indexes: список с названиями столбцов,
        первое значение должно быть названием столбца для комбинаций перебираемых значений.
        Последнее значение дожно быть названием столбца конечных значений функции.
        :param num_logic_var: количество логических переменных (у всех вариантов вроде 5)
        """
        self.indexes = indexes
        self.variants_index = indexes[0]
        self.result_index = indexes[-1]
        self.num_logic_var = num_logic_var
        self.table = self.make_table()
        self.kdnf_str = self.kdnf(self.table, self.variants_index, self.result_index)
        self.kknf_str = self.kknf(self.table, self.variants_index, self.result_index)

    @abstractmethod
    def make_table(self):
        pass

    @staticmethod
    def make_df_from_table(indexes, columns):
        """
        Метод который юзается для получения DataFrame из списков
        Он используется в конце каждого метода где нужно получить таблицу.
        Без приведения данных к фрейму записать их в csv не получится
        :param indexes: Список со строками названий столбцов
        :param columns: Список со списками данных
        :return:
        """
        return pd.DataFrame(
            {
                index: value for index, value in
                zip(
                    indexes,
                    [pd.Series(value, dtype=str) for value in columns]
                )
            }
        )

    @staticmethod
    def kdnf(df, variants_index, result_index) -> str:
        """
        Метод который генерит КДНФ.
        В строке для математических символов юзается символы Латеха
        В последней версии ворда есть возможность их юзать в формулах, остальным соболезную.
        В примере выводит строку в консольку
        :param df:
        :param variants_index:
        :param result_index:
        :return:
        """
        result_for_kdnf = df.loc[df[result_index] == '1']
        result = ''
        for _, row in result_for_kdnf.iterrows():
            variant = row[variants_index]
            result += (
                ''.join(
                    [
                        str('\\bar x_' + str(i+1)) if v == '0' else str('x_' + str(i+1))
                        for i, v in enumerate(variant)
                    ]
                ) + ' \\vee '
            )
        return result

    @staticmethod
    def kknf(df, variants_index, result_index) -> str:
        """
        Метод который генерит ККНФ. Генерятся лишние значки перед закрывающей скобкой, просто удаляйте их,
        чинить эту недоработку мне пока лень
        :param df:
        :param variants_index:
        :param result_index:
        :return:
        """
        result_for_kknf = df.loc[df[result_index] == '0']
        result = ''
        for _, row in result_for_kknf.iterrows():
            variant = row[variants_index]
            res = '('
            res += str(
                ''.join(
                    [
                        str('\\bar x_' + str(i+1) + ' \\vee ') if v == '1' else str('x_' + str(i+1) + ' \\vee ')
                        for i, v in enumerate(variant)
                    ]
                )
            )
            res += ')'
            result += res
        return result

    def write(self, name_file):
        self.table.to_csv(name_file)


class FirstStage(AbstractFirstStage):
    """
    Класс для получения таблицы истинности.
    Данные таблицы юзается в остальных классах, очень важно чтобы генерируемые данные были верные.
    Если не знаешь как их правильно сгенерить, то посмотри референс в файле test
    """
    def make_table(self):
        """
        Этот метод генерит данные для 9 варианта
        :return:
        """
        variants = [''.join(i) for i in itertools.product('01', repeat=self.num_logic_var)]  # x1x2x3x4
        column2 = [i[3]+i[4] for i in variants]  # x4x5
        column3 = [str(int(i, 2)) for i in column2]  # (x4x5)10
        column4 = [i[0] + i[1] + i[2] for i in variants]  # x1x2x3
        column5 = [str(int(i, 2)) for i in column4]  # (x1x2x3)10
        column6 = [str(int(first_num) + int(second_num)) for first_num, second_num in zip(column3, column5)]  # (x4x5)10 - (x1x2x3)10
        column7 = []  # f
        for second_num, res in zip(column5, column6):
            if int(second_num) == 1:
                column7.append('d')
            else:
                if (3 < int(res)) and (int(res) < 8):
                    column7.append('1')
                else:
                    column7.append('0')
        # эта часть кода с приведением данных к фрейму обязательна, не забудь при написании своего метода
        table = AbstractFirstStage.make_df_from_table(self.indexes, [
            variants,
            column2,
            column3,
            column4,
            column5,
            column6,
            column7
        ])
        return table


class Minimize:
    """
    Класс который состовляет таблицы для:
    поиска имликант (кубики),
    поиска значимых импликант (табличку в которой нужно раскрасить столбцы),
    поиска имплицент,
    поиска значимых имплицент
    """
    def __init__(self, first_stage: AbstractFirstStage):
        self.first_stage = first_stage
        self.table_truth = self.first_stage.table
        self.cube_table_implicant, self.table_essential_implicants = self.implicant()
        self.cube_table_implicent, self.table_essential_implicents = self.implicent()

    def implicant(self):
        """
        Метод который полностью разбирается с импликантами
        :return:
        """
        first_column = self.make_first_column(True)
        all_cubes = self.find_cubes(first_column)
        simple_implicants = self.make_max_cubes(all_cubes)
        cube_table = self.make_cubes_table(first_column, all_cubes, simple_implicants)
        table_essential_implicants = AbstractFirstStage.make_df_from_table(
            *self.essential_max_cubes(simple_implicants, True)
        )
        return cube_table, table_essential_implicants

    def implicent(self):
        """
        Метод который полностью разбирается с имплицентами
        :return:
        """
        first_column = self.make_first_column(False)
        all_cubes = self.find_cubes(first_column)
        simple_implicents = self.make_max_cubes(all_cubes)
        cube_table = self.make_cubes_table(first_column, all_cubes, simple_implicents)
        table_essential_implicents = AbstractFirstStage.make_df_from_table(
            *self.essential_max_cubes(simple_implicents, False)
        )
        return cube_table, table_essential_implicents

    def make_first_column(self, type_f: bool):
        """
        Метод для генерации первого столбца, в нем сортируются варианты логических
        переменных распределются по группам которые сортируются по количеству единиц
        Любое значение в столбце состоит из списка [
            'строка значения куба',
            [список строк из каких кубов сформирован этот куб],
            булевая_метка_которая_используется_для_определения_простых_импликант
        ]
        Т.к в этом столбце не кубы, то у значения список строк всегда пустой.
        :param type_f: для импликанты True, для имплицент False
        :return:
        """
        result_for_minimize = self.table_truth.query(f"f == '{str(int(type_f))}' | f == 'd'")
        result = []
        for i in range(self.first_stage.num_logic_var + 1):
            result_buf = []
            for _, row in result_for_minimize.iterrows():
                if str(row[self.first_stage.variants_index]).count('1') == i:
                    result_buf.append([row[self.first_stage.variants_index], [], False])
            if result_buf:
                result.append(result_buf)
        return result

    @staticmethod
    def find_cubes(first_column):
        """
        Мпетод для поиска кубов и формирования по ним значений столбцов
        :param first_column:
        :return:
        """
        all_columns_res = []
        previous_column = first_column
        while True:
            column_res = []
            c = 0  # счетчик для учета уже проверенных комбинаций
            for group_i, group in enumerate(previous_column):
                variants = {}
                if group_i == len(previous_column) - 1:
                    break
                for first_v_id, first_cube in enumerate(group):
                    first_cube_value = first_cube[0]
                    for second_v_id, second_cube in enumerate(previous_column[group_i + 1]):
                        second_cube_value = second_cube[0]
                        if sum([1 for a, b in zip(first_cube_value, second_cube_value) if a != b]) == 1:
                            first_cube[2] = True
                            second_cube[2] = True
                            buf_res = [
                                ''.join([
                                    'X' if a != b else a for a, b in zip(first_cube_value, second_cube_value)
                                ]),
                                [str(c + first_v_id + 1) + '-' + str(c + len(group) + second_v_id + 1)],
                                False
                            ]
                            find_cube_value = variants.get(buf_res[0])
                            if find_cube_value:  # обновляем список строк кубов из которых образуется значение
                                find_cube_value[1].append(buf_res[1][0])
                            else:
                                variants[buf_res[0]] = buf_res
                variants_values = variants.values()
                if variants_values:
                    column_res.append(list(variants_values))
                    c += len(group)
            if not column_res:
                break
            all_columns_res.append(column_res)
            previous_column = column_res
        return all_columns_res

    @staticmethod
    def make_max_cubes(all_cubes):
        """
        Метод для выделения максимальных кубов, все значения чья логическая метка равна False
        являются максимальными кубами
        :return:
        """
        z_res = []
        for column in all_cubes:
            for group in column:
                for value in group:
                    if not value[2]:
                        z_res.append(value[0])
        return z_res

    @staticmethod
    def make_cubes_table(first_column, all_cubes, max_cubes):
        """
        Метод который формирует таблицу всех кубов. Конец каждой группы выделяется значением '---',
        но в своей курсовой лучше выделять группы утолщеными границами таблицы.
        !!!Ни в коем случае не делайте в курсовой столбец 'Является макс. кубом - True False'!!!
        Если значение равно True, то это значение является простой импликантой/имплицентой и его нужно выделить
        выделяйте как угодно - жирным текстом, курсивом, цветом, плюсиком и т.д что хватит фантазии
        :param first_column:
        :param all_cubes:
        :param max_cubes:
        :return:
        """
        variants = []
        for group in first_column:
            variants.extend(group)
            variants.append(['---', '---', '---'])
        variants = pd.DataFrame({
            'K0(f)N(f)': pd.Series([variant[0] for variant in variants], dtype=str),
            'Является макс. кубом': pd.Series([not variants[2] for variants in variants], dtype=str)
        })
        columns_cubes = []
        for column in all_cubes:
            cubes = []
            for group in column:
                cubes.extend(group)
                cubes.append(['---', '---', '---'])
            columns_cubes.append(cubes)
        cubes_rows = [
            pd.DataFrame({
                f'K{num_column_cubes}(f)': pd.Series([cube[0] for cube in cubes], dtype=str),
                'Из каких кубов образуется': pd.Series([', '.join(cube[1]) for cube in cubes], dtype=str),
                'Является макс. кубом': pd.Series([not cube[2] for cube in cubes], dtype=str),
            }) for num_column_cubes, cubes in enumerate(columns_cubes)
        ]
        simple_implicants = pd.DataFrame({
            'Z(f)': pd.Series(max_cubes, dtype=str)
        })
        result = pd.concat([variants, *cubes_rows, simple_implicants], axis=1)
        return pd.DataFrame(result)

    def essential_max_cubes(self, max_cubes, type_f: bool):
        """
        Метод для формирования таблички макс кубов
        :param max_cubes:
        :param type_f: для импликанты True, для имплицент False
        :return:
        """
        table = []
        table_indexes = ['Макс. кубы']
        table.append(max_cubes)
        result_for_impl = self.table_truth.loc[self.table_truth[self.first_stage.result_index] == str(int(type_f))]
        for zero_cube in result_for_impl[self.first_stage.variants_index]:
            column = []
            table_indexes.append(zero_cube)
            for simple_implicant in max_cubes:

                if sum([1 for a, b in zip(zero_cube, simple_implicant) if (a != b) and (b != 'X')]) == 0:
                    column.append('*')
                else:
                    column.append(' ')
            table.append(column)
        return table_indexes, table

    @staticmethod
    def write(df, name_file: str):
        df.to_csv(name_file)


class CheckerFunction:
    """
    Класс который можно заюзать чтобы проверить правильность своей булевой функции
    Естественно правильность исходит из предположения правильности генерации таблицы истинности.
    Для реализации своей булевой функции нужно переопределить метод bool_func и в условии if
    прописать аналог своей булевой функции.
    Этот класс не более чем полезная необязательная плюшка
    """
    def __init__(self, first_stage: AbstractFirstStage):
        self.first_stage = first_stage
        self.table_truth = self.first_stage.table

    def check(self):
        result_for_check = self.table_truth.loc[self.table_truth[self.first_stage.result_index] == '1']
        for _, row in result_for_check.iterrows():
            res = self.bool_func(*[bool(int(i)) for i in row[self.first_stage.variants_index]])
            if not res:
                print(f'Ошибка в {row[self.first_stage.variants_index]}, получено {res}, вместо 1')
        result_for_check = self.table_truth.loc[self.table_truth[self.first_stage.result_index] == '0']
        for _, row in result_for_check.iterrows():
            res = self.bool_func(*[bool(int(i)) for i in row[self.first_stage.variants_index]])
            if res:
                print(f'Ошибка в {row[self.first_stage.variants_index]}, получено {res}, вместо 0')

    @staticmethod
    def bool_func(*args) -> bool:
        """
        Булевая функция выполняется в этом методе, для проверки своей функции его нужно изменить
        :param args:
        :return:
        """
        x1, x2, x3, x4, x5 = args
        if (x1 and (not((x3 and x5) or (x2 and x4)) or (not(x2) and not(x4)))) or (not(x1) and ((x3 and x5) or (x2 and x4))):
            return True
        else:
            return False


def check():
    """
    Пример использования проверки булевой функции
    :return:
    """
    first_stage = FirstStage(
        ['x1x2x3x4x5', 'x4x5', '(x4x5)10', 'x1x2x3', '(x1x2x3)10', '(x4x5+x1x2x3)10', 'f'], 5)
    checker = CheckerFunction(first_stage)
    checker.check()


def main():
    """
    Пример генерации и записи в csv всех возможных таблиц для 9 варианта
    :return:
    """
    first_stage = FirstStage(
        ['x1x2x3x4x5', 'x4x5', '(x4x5)10', 'x1x2x3', '(x1x2x3)10', '(x4x5+x1x2x3)10', 'f'], 5)
    first_stage.write('truth_table.csv')

    print('КДНФ:', first_stage.kdnf_str)
    print('ККНФ:', first_stage.kknf_str)

    minimize = Minimize(first_stage)
    minimize.write(minimize.cube_table_implicant, 'cube.csv')
    minimize.write(minimize.table_essential_implicants, 'essential_implicants.csv')

    minimize.write(minimize.cube_table_implicent, 'cube_implicent.csv')
    minimize.write(minimize.table_essential_implicents, 'essential_implicents.csv')


if __name__ == '__main__':
    main()
