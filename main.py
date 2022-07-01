import numpy as np
import matplotlib.pyplot as plt


# создание  трехмерной сетки
def create_3d_grid(x0,y0,t0,x_max,y_max,t_max,dx,dy,dt):
    x = np.arange(x0, x_max, dx)
    y = np.arange(y0, y_max, dy)
    t = np.arange(t0, t_max, dt)
    X, Y, T = np.meshgrid(x, y, t)
    return X,Y,T

#процудура по 3-д сетке строит 2д-сетку.
# Значение в соответстующей клетке массива - индекс времени когда p_grid максимальна.
# Не само время а его номер
def t_argmax_values(p_grid):
    n=len(p_grid)# по y
    m=len(p_grid[0])#x
    q=np.zeros([len(p_grid),len(p_grid[0])])
    for i in range(n):
        for j in range(m):
            q[i,j]=np.argmax(p_grid[i,j,:])
    #создадим список gridpoint
    grid_types = []
    for j in range(n):
        grid_types.append([0]*m)
        for i in range(m):
            grid_types[j][i]='far'

    return q,grid_types

#процедура добавления точки с индексами коорлинат (i,j) to considered
#def add_point_to_considered(i,j,grid_types):
#    grid_types[j][i]='considered'

def add_three_neib_to_considere(i,j,ii,jj,grid_types):
    if grid_types[j][ii] =='far':
        grid_types[j][ii] = 'considered'
    if grid_types[jj][ii] =='far':
        grid_types[jj][ii] = 'considered'
    if grid_types[jj][i] =='far':
        grid_types[jj][i] = 'considered'


#процедура добавления точки с индексами коорлинат (i,j) to accepted
def add_point_to_accepted(i,j,grid_types):
    n=len(grid_types)
    m=len(grid_types[0])
    grid_types[j][i]='accepted'
    if i<m-1 and j<n-1:
        ii=i+1
        jj=j+1
        add_three_neib_to_considere(i, j, ii, jj, grid_types)
    if i>0 and j>0:
        ii=i-1
        jj=j-1
        add_three_neib_to_considere(i, j, ii, jj, grid_types)
    if i>0 and j<n-1:
       ii=i-1
       jj=j+1
       add_three_neib_to_considere(i, j, ii, jj, grid_types)
    if i<m-1 and j>0:
        ii=i+1
        jj=j-1
        add_three_neib_to_considere(i, j, ii, jj, grid_types)


def sgn(x):
    if x>0:
        s=1
    if x<0:
        s=-1
    if x==0:
        s=0
    return s

#процудура вычисляющая кумулятивное время если прийти из (ii,jj) in (i,j)
#(ii,jj) - accepted neghbor point
def calculete_v(i,j,ii,jj,cum_times,q_times):
    #v = cum_times[jj][ii][0] + abs(q_times[jj, ii] - q_times[j, i])
    v = abs(q_times[jj, ii] - q_times[j, i])

    return v,sgn(q_times[j, i] - q_times[jj, ii])


def add_v_for_accepted_neighbor(i,j,ii,jj,grid_types, cum_times, q_times,v_list,Penalty):
    if grid_types[jj][ii]=='accepted':
        v, sg = calculete_v(i, j, ii, jj, cum_times, q_times)
        v_list.append([v, sg,v+sg*(sg-1)*Penalty,ii,jj])

#запускает add_v_for_accepted_neighbor для трех соседнихъ точек
def add_v_for_three_accepted_neighbor(i, j, ii, jj, grid_types, cum_times, q_times, v_list,Penalty):
    add_v_for_accepted_neighbor(i, j, ii, jj, grid_types, cum_times, q_times, v_list,Penalty)
    add_v_for_accepted_neighbor(i, j, i, jj, grid_types, cum_times, q_times, v_list,Penalty)
    add_v_for_accepted_neighbor(i, j, ii, j, grid_types, cum_times, q_times, v_list,Penalty)

#процедура вычисления cumulative_times для considered точки (i,j)
#cum_times--матрица уже посчитанных cum_times
#q-times --- матрица значений индексов времени когда плотность в этой точке максимальна
def calculete_considered_cum_time(i,j,grid_types,cum_times,q_times,Penalty):
    n = len(grid_types)
    m = len(grid_types[0])
    v_list=[]
    #v_value_list=[]
    if i < m-1 and j < n-1:
        ii = i + 1
        jj = j + 1
        add_v_for_three_accepted_neighbor(i, j, ii, jj, grid_types, cum_times, q_times, v_list, Penalty)
    if i > 0 and j > 0:
        ii = i - 1
        jj = j - 1
        add_v_for_three_accepted_neighbor(i, j, ii, jj, grid_types, cum_times, q_times, v_list, Penalty)
    if i > 0 and j < n-1:
        ii = i - 1
        jj = j + 1
        add_v_for_three_accepted_neighbor(i, j, ii, jj, grid_types, cum_times, q_times, v_list, Penalty)
    if i < m-1 and j > 0:
        ii = i + 1
        jj = j - 1
        add_v_for_three_accepted_neighbor(i, j, ii, jj, grid_types, cum_times, q_times, v_list, Penalty)
    #создадим отдельно список значений v
    v_value_list = np.zeros(len(v_list))
    for i in range(len(v_list)):
        v_value_list[i]=v_list[i][2]
    arm=np.argmin(v_value_list)
    V=v_value_list[arm]
    previous_x=v_list[arm][3]
    previous_y=v_list[arm][4]

    return V,previous_x,previous_y#,v_list,v_value_list

#процедура, вычислаяюща cum_time во всех considered точках и возвращающая considered point с минимальным cum_time
def min_cum_t_considered_point(grid_types,cum_times,q_times,Penalty):
    n = len(grid_types) # y
    m = len(grid_types[0])#x
    v_considered=[]

    for i in range(m):
        for j in range(n):
            if grid_types[j][i]=='considered':
                v,prx,pry=calculete_considered_cum_time(i, j, grid_types, cum_times, q_times, Penalty)
                v_considered.append([v,i,j,prx,pry])
    k=len(v_considered)
    v_value_considered = np.zeros(k)
    for i in range(k):
        v_value_considered[i]=v_considered[i][0]
    arm=np.argmin(v_value_considered)
    point_x=v_considered[arm][1]
    point_y=v_considered[arm][2]
    previous_point_x=v_considered[arm][3]
    previous_point_y = v_considered[arm][4]
    V=v_value_considered[arm]

    return V,point_x,point_y,previous_point_x,previous_point_y


#процудура  длобавляющая  одной точки в accepted
def add_to_accepted(grid_types,cum_times,q_times,Penalty):
    print('add_to_accepted')
    V, point_x, point_y, previous_point_x, previous_point_y=\
        min_cum_t_considered_point(grid_types,cum_times,q_times,Penalty)
    print('point',point_x, point_y)
    print('value V',V)
    print('previous point',previous_point_x, previous_point_y)

    add_point_to_accepted(point_x, point_y, grid_types)
    cum_times[point_y][point_x][0]=V
    cum_times[point_y][point_x][1]=previous_point_x
    cum_times[point_y][point_x][2]=previous_point_y
    cum_times[point_y][point_x][3]=cum_times[previous_point_y][previous_point_x][3]+1


#процедура добавления всех точек в accrpted
def calculete_cum_times(grid_types,cum_times,q_times,Penalty):
    n = len(grid_types)  # y
    m = len(grid_types[0])  # x
    #point_number=0# номер под которым ячейка добавляется к accepted
    #посчитаем количество accepted точек
    k=0
    for i in range(m):
        for j in range(n):
            if grid_types[j][i]=='accepted':
                k+=1
    if k==0:
        print('there are no accepted points')
    else:
        for i in range(m*n-k):
            add_to_accepted(grid_types, cum_times, q_times, Penalty)
            #point_number+=1
            print('gridtypes',grid_types)
            print('cumtimes',cum_times)

#процедура нахождения наискорейшего(с минимальной разницей значений) пути через клетку
def fastext_path_through_the_cell(t_ld,t_rd,t_ru,t_lu):
    #t_ld,t_rd,t_ru,t_lu - значения времени максимальной концентрации в углах (левый нижний, правый нижний и т.д. ячейки)
    corners=['ld','rd','ru','lu']
    t=np.array([t_ld,t_rd,t_ru,t_lu])
    path=[]
    weigts=[]
    for i in range(4):
        for j in range(i+1,4):
            w=abs(t[i]-t[j])
            path.append([i,j,corners[i],corners[j],w])
            weigts.append(w)
    fastetst_time=np.argmin(weigts)
    if t[path[fastetst_time][0]]>t[path[fastetst_time][1]]:
        start=path[fastetst_time][2]
        finish=path[fastetst_time][3]
    else:
        start = path[fastetst_time][3]
        finish= path[fastetst_time][2]

    return path,start,finish


def draw_trajectory(x0,y0,dx,dy,x_max,y_max,cum_times):
    # нарисуем сетку
    FirstCoordinate = []
    SecondCoordinate = []
    x = np.arange(x0, x_max, dx)
    y = np.arange(y0, y_max, dy)
    for i in range(len(x)):
        for j in range(len(y)):
            FirstCoordinate = FirstCoordinate + [i]
            SecondCoordinate = SecondCoordinate + [j]
    # print('coord', FirstCoordinate, SecondCoordinate)
    plt.scatter(FirstCoordinate, SecondCoordinate, color='black')

    m = len(x)
    n=len(y)
    trajectory_numbers = np.zeros([n, m])
    for i in range(m):
        for j in range(n):
            trajectory_numbers[j][i] = cum_times[j][i][3]
    armax = np.argmax(trajectory_numbers)

    # m=len(x)

    i = armax % m
    j = armax // m

    while cum_times[j][i][3] > 0:
        ii = cum_times[j][i][1]
        jj = cum_times[j][i][2]
        plt.scatter([i], [j], color='red')
        plt.scatter([ii], [jj], color='green')
        plt.plot([i, ii], [j, jj])
        i = int(ii)
        j = int(jj)


#для примера
def p(x,y,t):
    p=t*(x-1)*(y-1.5)
    return p
def main():
    x0 = 0
    y0 = 1
    t0 = 0
    dx = 1
    dy = 0.2
    dt = 0.4
    x_max = 5
    y_max = 2
    t_max = 0.5#1
    X,Y,T=create_3d_grid(x0,y0,t0,x_max,y_max,t_max,dx,dy,dt)

    path,start,finish=fastext_path_through_the_cell(1,2,3,1)


    #построим сетку из точек сетки
    P=p(X, Y, T)
    q_times,grid=t_argmax_values(P)
    q_times[0][1]=1
    #q_times=
    print('grid',grid)
    add_point_to_accepted(1, 0, grid)
    print('grid', grid)
    #q_times=np.array([[3,2,1],[5,3,4]])
    print('qtimes',q_times)

    #
    Penalty=3
    cum_times=np.zeros([len(X),len(X[0]),4])#массив в каждой ячейке хроанится cum_time , номер предыдущей ячейки [1,2]
                                            # и длина траектории с концом в этой точке
    for i in range(len(X)):
        for j in range(len(X[0])):
            cum_times[i][j][1] = -1
            cum_times[i][j][2] = -1


    print('cumtimes',cum_times)

    #calculete_cum_times(grid, cum_times, q_times, Penalty)
    calculete_cum_times(grid, cum_times, q_times, Penalty)

    #нарисуем траекторию
    draw_trajectory(x0,y0,dx,dy,x_max,y_max,cum_times)



    print('ij', i, j)


    print('ij',i,j)
    print('cum_times[i][j][1]',cum_times[1][1][3])


    plt.show()

    #выделим самую длинную траекторию


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
