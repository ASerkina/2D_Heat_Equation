#include <iostream>
#include <math.h>
#include <fstream>

int main()
{
    // переменные и константы
    const int m = 100;
    int k = 0, l = 0;
    double temp;
    double** T1 = new double* [m];
    double** T2 = new double* [m];
    for (int i = 0; i < m; ++i)
    {
        T1[i] = new double[m];
        T2[i] = new double[m];
    }
    const double h = 1.0f / (m - 1);
    const double t = h;
    const double a = -1.0f, b = -1.0f * a / 3;
    // x = 2y - 1
    // y = (x + 1) / 2
    double eps = 1.0f;
    double max = 1.0f;
    double stop = 1.0f;
    std::ofstream output("log2.txt");

    // начальные условия
    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            T1[i][j] = 0;
            T2[i][j] = 0;
        }
    }

    // непосредственно процесс
    while (stop > 0.00001)
    {
        k = 0;
        max = 1.0f;
        while (max > 0.00001)
        {
            // граничные условия
            for (int i = 0; i < m; i++)
            {
                T2[i][0] = h* b + T2[i][1];
                T2[i][m - 1] =  h*b + T2[i][m - 2];
            }
            for (int j = 0; j < m; j++)
            {
                T2[0][j] = h* b + T2[1][j];
                T2[m - 1][j] = h*a + T2[m - 2][j];
            }

            // внутренние температуры
            for (int i = 1; i < m - 1; i++)
                for (int j = 1; j < m - 1; j++)
                    T2[i][j] = (t * (T2[i + 1][j] + T2[i - 1][j] + T2[i][j + 1] + T2[i][j - 1]) + h * h * T1[i][j]) / (h * h + 4 * t);
            if (k % 10 == 0)
            {
                max = 0.0f;
                for (int i = 1; i < m - 1; i++)
                    for (int j = 1; j < m - 1; j++)
                    {
                        eps = abs(T2[i][j] - (t * (T2[i + 1][j] + T2[i - 1][j] + T2[i][j + 1] + T2[i][j - 1]) + h * h * T1[i][j]) / (h * h + 4 * t));
                        if (max < eps)
                            max = eps;
                    }
            }
        }
        // if (l % 3 == 0)
        // {
        for (int i = 0; i < m; ++i)
        {
            for (int j = 0; j < m; ++j) {
                temp = (T2[i][j] + 1.0) / 2;
                output << temp << ' ';
            }
            output << std::endl;
        }
        // output << std::endl;
    // }
        stop = 0.0f;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < m; ++j)
            {
                if (stop < abs(T2[i][j] - T1[i][j]))
                    stop = abs(T2[i][j] - T1[i][j]);
            }
        for (int i = 0; i < m; i++)
            for (int j = 0; j < m; j++)
            {
                temp = T1[i][j];
                T1[i][j] = T2[i][j];
                T2[i][j] = temp;
            }
        ++l;
    }

    // зачистка
    output.close();
    for (int i = 0; i < m; ++i)
        delete[] T1[i], T2[i];
    delete[] T1, T2;

    return 0;
}