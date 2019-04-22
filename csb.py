# Repo: playground-yr07d3ss
# 在前一版的優良基礎上，改成像 DarthBoss 一樣，不出賤招，直接比試操控能力 
# ---> 約 360 名。比略出賤招的 250 名略遜。eval_to_cp1() 往前多推算 1 步打分即可。

import sys
import math
import numpy as np

# -------------------------------------------------------------------------
# 17:20 2019-04-15 照 Magus 的建議改寫成 OOP 

class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y
    def distance2(self, p):
        # distance^2 from this point to p
        return (this.x - p.x)**2 + (this.y - p.y)**2
    def distance(self, p):
        # from this point to p
        return sqrt(self.distance2(p))
    def closest(self, a, b):
        # from this point to line(Point a, Point b)

class Unit(Point):
    def __init__(self, p0, id, r, v0):
        self.x, self.y, self.id, self.r, self.vx, self.vy = p0.x, p0.y, id, r, v0.x, v0.y
    def collision(self, u):
        pass
    def bounce(self, u):
        pass

class Checkpoint(Unit):
    def bounce(self, u):
        pass

class Pod(Unit):
    angle, nextCheckpointId, checked, timeout, partner, shield
    def __init__(self, p0, id, r, v0):
        self.x, self.y, self.id, self.r, self.vx, self.vy = p0.x, p0.y, id, r, v0.x, v0.y

    def getAngle(self, p):
        d = self.distance(p);
        dx = (p.x - this.x) / d;
        dy = (p.y - this.y) / d;

        # Simple trigonometry. We multiply by 180.0 / PI to convert radiants to degrees.
        a = math.acos(dx) * 180.0 / math.pi;

        # If the point I want is below me, I have to shift the angle for it to be correct
        if (dy < 0):
            a = 360.0 - a;
        return a;
        
        
    def diffAngle(self, p):
        a = self.getAngle(p);

        # To know whether we should turn clockwise or not we look at the two ways and keep the smallest
        # The ternary operators replace the use of a modulo operator which would be slower
        right = a - self.angle if self.angle <= a else 360.0 - self.angle + a
        left  = self.angle - a if self.angle >= a else self.angle + 360.0 - a;

        if (right < left):
            return right;
        else:
            # We return a negative angle if we must rotate to left
            return -left;

    def rotate(self, p):
        a = this.diffAngle(p);

        # Can't turn by more than 18 degree in one turn
        if (a > 18.0):
            a = 18.0;
        elif (a < -18.0):
            a = -18.0;

        self.angle += a;

        # The % operator is slow. If we can avoid it, it's better.
        if (self.angle >= 360.0):
            self.angle = self.angle - 360.0;
        elif (self.angle < 0.0):
            self.angle += 360.0;

    def boost(self, thrust):
        # Don't forget that a pod which has activated its shield cannot accelerate for 3 turns
        if (self.shield):
            return;

        # Conversion of the angle to radiants
        ra = self.angle * math.pi / 180.0;

        # Trigonometry
        self.vx += math.cos(ra) * thrust;
        self.vy += math.sin(ra) * thrust;


    def move(self, t):
        # t will be useful later when we'll want to simulate an entire turn while taking into account collisions. 
        self.x += self.vx * t;
        self.y += self.vy * t;

    def end(self):
        self.x = round(self.x);
        self.y = round(self.y);
        self.vx = int(self.vx * 0.85);
        self.vy = int(self.vy * 0.85);

        # Don't forget that the timeout goes down by 1 each turn. It is reset to 100 when you pass a checkpoint
        self.timeout -= 1;

    def play(self, p, thrust):
        self.rotate(p);
        self.boost(thrust);
        self.move(1.0);
        self.end();

    def bounce(self, u):
        pass
    def output(self, move):
        pass
        
class Solusion:
    def randomize(self):
        pass
    
class Move:
    def mutate(self, amplitude)
        pass
        
class Collision:
    unit a, unit b, float t
    
    
    
    
    
# -------------------------------------------------------------------------

step = 0
pod_history = [[0],[0]] # History checkpoint idx of pod0,pod1
opp_history = [[0],[0]] # History checkpoint idx of opp0,opp1
# blocker_last_distance = list((9999,)*3) # pod0,opp0,opp1 紀錄與其他三人的前一距離，判斷是否接近中。
max_thrust = 200
cp_spot = 590

# formulas
def round(v0):
    # 四捨五入到整數
    # return int(n+0.5)
    # return (uv(v0)*(norm(v0)+0.5).astype(int)).astype(int)
    return np.round(v0,0)

def v(x,y):
    # vectorize to numpy array which is also a vector 
    return np.array([x,y])

def uv(v):
    # Normalize the given vector to an unit vector
    # https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
    norm = np.linalg.norm(v)
    return v if norm==0 else v/norm;

def norm(v):
    # length of a vector, where v (vector) is a np.array
    return np.linalg.norm(v)

def normalized_angle(degrees):
    # https://stackoverflow.com/questions/2320986/easy-way-to-keeping-angles-between-179-and-180-degrees
    return ((degrees + 179) % 360 + 360) % 360 - 179;

def absolute_angle(vec):
    # input: a vector
    # output: absolute angle to x axis
    # 檢查 absolute_angle(v) 的輸出：
    # results = [absolute_angle(random.randint(-100,100),random.randint(-100,100)) for i in range(10000)]
    # results = [ i for i in results if (i >= 180 or i <= -180) ]
    # 所有的結果都在 -180 ~ 180 之間，且只會有 180 而沒有 -180, 非常理想。
    v0 = v(1,0)  # unit vector on x axis 
    rad = np.math.atan2(np.linalg.det([v0,vec]),np.dot(v0,vec))
    return np.degrees(rad)  # .astype(int)

def angle2uv(an):
    # Convert the given absolute angle in degrees to a unit vector that represents the angle
    an1 = np.deg2rad(an)
    return uv(v(math.cos(an1), math.sin(an1))) 

def perpendicular(v0):
    # Get two perpendicular unit vectors of the given one
    # https://gamedev.stackexchange.com/questions/70075/how-can-i-find-the-perpendicular-to-a-2d-vector/113394
    return uv(v(-v0[1],v0[0])), uv(v(v0[1],-v0[0]))

def intersection_angle(v0, v1):
    # intersection angle (-180 ~ 180) in degrees from v0 to v1
    rad = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1))
    return np.degrees(rad)  # .astype(int)

def angle2target(an, p0, pt):
    # output: intersection angle (-180 ~ 180) from the pod to target point 
    # input: abslute angle of the pod, pod position, target position
    v0 = angle2uv(an)  # a vector points to the pod
    v1 = pt - p0  # v(tx, ty) - v(x, y)
    return intersection_angle(v0, v1)
    
def distance(v0, v1):
    # input: pod position, next checkpoint position
    # output: distance in float , always positive
    return np.linalg.norm(v1-v0)

def decel(v0):
    # Deceleration Vector decel(v0) is the Resistance or Friction against V0 and Thrust
    # v1 = v0 + decel(v0) + thrust + decel(thrust); Where thrust is a vector
    aV0 = norm(v0)
    if 787 < aV0:
        a = 0.000003422507988963333
        b = -0.15933880049599397
        c = 5.613010798946107
    elif 346 < aV0 <= 787:
        a = -2.400823194257034e-7
        b = -0.15091118442691628
        c = -0.35617460514862387
    elif 152 < aV0 <= 346:
        a = 0.00012932973459289256
        b = -0.21738641475483583 
        c = 8.110644399460199
    elif 78 < aV0 <= 152:
        a = 0.0005754475703324807
        b = -0.2921355498721227
        c = 8.005882352941171
    elif 39 < aV0 <= 78:
        a = -0.0013550135501355016
        b = 0.028455284552845555
        c = -6.344173441734412
    if aV0 <= 39:
        if 28 < aV0 <= 39:
            a = (-1/6)
            b = 0.5
        elif 9 < aV0 <= 28:
            a = -0.2
            b = 0.6
        elif 5 < aV0 <= 9:
            a = 0
            b = -2
        else:    
            a = 0
            b = -1
        aV0r = a*aV0 + b  # Regression 
    else:
        aV0r = a*aV0**2 + b*aV0 + c  # Regression 
    return uv(v0)*aV0r if aV0 > abs(aV0r) else -v0

def next_angle18(an0,p0,pt):
    # 下一個朝向的絕對角度 absolute angle 每次最多轉 +-18度。
    a2t = angle2target(an0,p0,pt)  # 朝向 pt 的夾角
    result = an0 + (min(a2t,18) if a2t > 0 else max(a2t,-18)) # 相加造成超過 180 度，需要 normalize
    return normalized_angle(result);

def best_thrust(p0,v0,an,pt):
    an1 = angle2uv(next_angle18(an,p0,pt))
    p1 = p0 + v0 
    a = 0
    b = max_thrust
    pa = p1 + an1 * a
    pb = p1 + an1 * b

    for i in range(9): # 2^8 = 256 that is enough for max_thrust 200
        if distance(pa,pt) >= distance(pb,pt):
            a1 = int((a+b)/2)
            if a1 == a or a1 == b:
                # b is the best 
                return b
            else:
                a = a1
                pa = p1 + an1 * a
        else:
            b1 = int((a+b)/2)
            if b1 == a or b1 == b:
                # a is the best 
                return a
            else:
                b = b1
                pb = p1 + an1 * b
    raise Exception(f"Strange situation in best_thrust a={a} b={b} pa={pa} pb={pb} pt={pt}")

def enough_to_spot(p0, v0, cp0, steps, tolerance):
    # present position and speed p0, v0; target position cp0; foresee steps; spot tolerance
    if norm(v0) <= 10: return False  # 速度 0 會造成誤會 to this algorithm, add some tolerance to 10.
    if steps < 0: raise Exception("Strange steps: " + str(steps)) # No steps 
    if steps == 0: return True if distance(p0, cp0) <= tolerance else False 
    future_positions = []  # 靠慣性，掉頭完成前，未來的位置序列
    future_distances = [distance(p0, cp0).astype(int)]
    for i in range(steps):
        p0 = p0 + v0
        future_positions.append(p0.astype(int))
        v0 = v0 + decel(v0)
    future_distances += [distance(p, cp0).astype(int) for p in future_positions] # 未來的距離序列
    closest = np.argmin(future_distances)
    return (closest != steps) and (future_distances[closest] <= tolerance) # 慣性夠不夠滑向目標

# 把 input 的 thrust 改成 tuple (x,y,thrust) 比較自然, tp 就是 target point vector 用來算餘程距離。
def move(p0=v(0,0),v0=v(0,0),an0=v(0,0),thrust=(0,0,max_thrust),tp=v(0,0)):
    # 準確算出走完本 step 留給下一個 step 的參數，精準！
    # Input: p0 起點位置, v0 速度, an0 角度, tp 目標位置用來算餘程，以上都是向量; 
    #        thrust 是 (x,y,推力) tuple。
    an1 = next_angle18(an0,p0,v(*thrust[:2]))  # 實際下個朝向 (並非 thrust 的方向，要打折扣)
    force = angle2uv(an1)*thrust[2]
    v1 = v0 + decel(v0) + force + decel(force)
    p1 = p0 + v0 + force
    return tuple([i.astype(int) for i in map(round,(p1,v1,an1,distance(p1,tp)))])
    # (p1, v1, an1, distance to tp)

def arrive_soon(p0, v0, an0, cp0):
    # 檢查全速往前衝 10 步「會不會到？」
    # all are vectors
    flag = False
    # bigest_v0 = 0
    for i in range(10): # [ ] 可能應該用 9步預估，10步好像會有失誤。
        p0,v0,an0,dist = move(p0,v0,an0,(*tuple(cp0),max_thrust),cp0)
        # bigest_v0 = v0 if norm(v0) > norm(bigest_v0) else bigest_v0
        if dist <= cp_spot: 
            flag = True
            break
    return flag # , bigest_v0

def good_direction(p0,v0,cp0,cp_spot):
    # This is a practice of getting 'dot to line' distance  
    return abs(np.cross(v0,cp0-p0)/np.linalg.norm(v0)) <= cp_spot 

deceleration_table  = [1127, 957, 813, 691, 587, 498, 423, 359, 305,
    259, 220, 187, 158, 134, 113, 96, 81, 68, 57, 48, 40, 34, 28, 23, 19, 16, 13,
    11, 9, 7, 5, 4, 3, 2, 1, 0]
def inertia(v0, p0, pt):
    # 傳回值是個很大的向量，不用管它，我們要的是它的「方向」
    # 照目前慣性，將來會滑到哪裡。計算 pt 方向時 pt(的方向) = (pt -inertia(v0))的方向
    # 但是 |inertia| 不能超過 |pt - p0| 要調整它的長度，否則可能變成轉向。
    length = distance(pt,p0) 
    aV0 = norm(v0)
    for i in range(len(deceleration_table)):
        if deceleration_table[i] <= aV0:
            break
    total_inertia = v0 + uv(v0)*sum(deceleration_table[i:])
    return uv(total_inertia)*length if (norm(total_inertia)+1) >= length else total_inertia  # +1 取一點 tolerance:

def best_to_cp1(p0, v0, an0, cp0, cp1):
    # 前一版是『嘗試盡量用力』預轉向 cp1，傳回 (bool, pt, thrust) 
    # 這一版是『最佳用力』預轉向 cp1，傳回 (bool, pt, thrust) 
    bb = fastest_to_cp1(p0, v0, an0, cp0, cp1)
    if not bb[0]: return bb  # 這時的 pt,thrust 都沒有意義
    pt1, a, b = bb[1], 0, bb[2]  # 角度固定，a,b 都行，找出中間最好的。有可能 b 一來就是 0！
    # 特性：
    #     thrust a 從必定成功的 0 開始往上嘗試
    #     thrust b 從最大的 fastest_to_cp1() [以前的 turn_to_cp1()] 開始往下嘗試
    #     兩頭都必定成功，但要找出最好的。這是本 function 上場的時候了。 
    for binary in range(10):  # |a-b| < 2^8 一定可以在十次以內找到。
        if abs(a-b)<=1: # 當 a b 相等或相鄰就到底了，答案就是 b 取快一點的。
            return True, pt1, b
        la = eval_to_cp1(p0,v0,an0,pt1,a,cp0,cp1)  # a thrust 的分數
        lb = eval_to_cp1(p0,v0,an0,pt1,b,cp0,cp1)  # b thrust 的分數
        if la < lb:  
            # 更好的 thrust 在下半段，把 b 調下來
            b = int((a+b)/2)
        else:
            # 更好的 thrust 在上半段，把 a 調上去
            a = int((a+b)/2)
    raise Exception(f"Strange situation in best_to_cp1() {a}, {b}, {p0}, {v0}, {an0}, {cp0}, {cp1}, {pt1}")

def eval_to_cp1(p0, v0, an0, pt1, force, cp0, cp1): 
    # 評估 force, 預轉過 cp0 之後與 cp1 的距離
    # 第一步用要接受考驗的 pt1,force 試跑
    p0,v0,an0,dist = move(p0,v0,an0,(*tuple(pt1),force),cp0)
    
    # /* 隨後到達 cp0 之前用 fastest_to_cp1() 試跑 --> Timeout :-( */
    # 簡化成：直接把 v0 平移到 cp0 的中心點，當作已經達陣。 那也就是把 cp0 直接當成 p0 就對了。
    p0 = cp0

    # 到達 cp0 之後用 best_thrust() 試跑
    # 1次:360 2次:由3次的377兩小時進步到356 3次:377 5次:沒更好 10次:timeout
    p0,v0,an0,dist = move(p0, v0, an0, (*tuple(cp1),best_thrust(p0,v0,an0,cp1)), cp1)
    
    # 傳回與 cp1 的距離，小的贏。
    return dist

def fastest_to_cp1(p0, v0, an0, cp0, cp1):
    # 嘗試盡量用力「預轉」向 cp1，傳回 (bool, pt, thrust) 
    # 當 False 時表示根本滑不到，要另外重新對準方向。
    if not enough_to_spot(p0, v0, cp0, 10, cp_spot):
        return False, v(-1,-1), -1  # 這時的 pt,thrust 都沒有意義
    a, b = 0, max_thrust # a,b 兩極之間一開始 a=0 時直衝是最簡單的猜測值。
    pt1 = round(cp1-inertia(v0,p0,cp1)).astype(int)  # 指向 cp1 的修正方向
    for i in range(10):  # max_thrust 200 < 2^8 但是已知 8 不夠，乾脆用 10 算了。 好像 a,b 相鄰時要多做一次。
        p1,v1,an1,dist = move(p0,v0,an0,(*tuple(pt1),b),cp0)  # <--- b 是第一志願
        if dist <= cp_spot: 
            return True, pt1, b
        else:
            if enough_to_spot(p1, v1, cp0, 9, cp_spot): # 第一步上面走過了， worst case 180度迴轉剩 9步。
                return True, pt1, b
        # 來到這裡表示上面的 b 不成功，而 a 是會成功的，故取中間值。這個值不是第一志願，即使成功也要繼續往上嘗試。
        if b == a+1: # 當 a b 相鄰而 b 是失敗的，答案就是 a 了
            return True, pt1, a  # 否則往下走就是不停取 a = a1 來嘗試的 infinit loop, 因為 b 已經失敗。
        a1 = int((a+b)/2)
        p1,v1,an1,dist = move(p0,v0,an0,(*tuple(pt1),a1),cp0)  # <--- 稱為 a1 因為不是第一志願，希望能把 a 調上來。
        a1_is_good = False
        if dist <= cp_spot: 
            a1_is_good = True
        else:
            if enough_to_spot(p1, v1, cp0, 9, cp_spot): # 第一步上面走過了，完成迴轉剩 9步。
                a1_is_good = True
        if a1_is_good:
            a = a1  # 目標在 b ~ a1 之間，其中 a1 已知成功，相當於 a 故取代 a 整個重來。
        else:
            # 當 a1 不成功時，答案在 a1-1 ~ a 之間，而 a1-1 相當於新的 b 點故取代之，整個重來。
            b = max(a1-1, 0)  # 用 max() 買個保險，不知會不會發生。
    raise Exception(f"Strange situation in turn_to_cp1() a={a} b={b}, {p0}, {v0}, {an0}, {cp0}, {cp1}")
    # 第一關經 enough_to_spot() 篩選之後能進來的必定成功，否則就很奇怪了
    
# -------------------------------------------------------------------------

def collision_detect_pod0(thrust,next_positions):
    if (distance(v(*next_positions[0]),v(*next_positions[2])) < 800 
       or distance(v(*next_positions[0]),v(*next_positions[3])) < 800):
       thrust = 'SHIELD'
    return thrust

def collision_detect_pod1(thrust,next_positions):
    if distance(v(*next_positions[0]),v(*next_positions[1])) < 1000:
        thrust = 0 # yield
    if (distance(v(*next_positions[1]),v(*next_positions[2])) < 800 
       or distance(v(*next_positions[1]),v(*next_positions[3])) < 800):
        thrust = 'SHIELD'
    return thrust

# -------------------------------------------------------------------------

laps = int(input())
checkpoint_count = int(input())
cp = [];
for i in range(checkpoint_count): cp.append([int(j) for j in input().split()])

# game loop
while True:
    pod = [] # my pods, reset every run to reload data
    opp = [] # opponent's pods, reset every run to reload data
    for i in (0,1):
        pod.append(dict(zip(['x', 'y', 'vx', 'vy', 'an', 'id'],[int(j) for j in input().split()])))
    for i in (0,1):
        opp.append(dict(zip(['x', 'y', 'vx', 'vy', 'an', 'id'],[int(j) for j in input().split()])))
        
    # 紀錄前一次的目標位置。可用來分辨目標到了沒，到達之前不會變。
    # 亦可用來分辨 opp 是 striker 還是 blocker.
    for i in (0,1): # pod0,1  opp0,1
        if pod_history[i][-1] != pod[i]['id']:
            pod_history[i].append(pod[i]['id'])
            if i==0: pre_turn = False  # my striker 達陣了，終止「預轉」狀態
        if opp_history[i][-1] != opp[i]['id']:
            opp_history[i].append(opp[i]['id'])

    for i in (0,1):
        # read data
        x, y, vx, vy, an, idx = pod[i].values();
        nx, ny = cp[idx]  # next checkpoint
        nnx, nny = cp[0] if (idx + 1) >= len(cp) else cp[idx+1]  # next next checkpoint nnp
        # laps -= 1 if idx == 0 else 0  # 最後一圈 laps==1 時就不要「熄火調頭策略」了
        
        p0 = v(x,y)
        v0 = v(vx,vy)
        cp0 = v(nx,ny)
        cp1 = v(nnx,nny)

        pt = cp0 if arrive_soon(p0, v0, an, cp0) else round(cp0-inertia(v0,p0,cp0)).astype(int)
        thrust = best_thrust(p0, v0, an, pt)
        
        pre_turn, direction, force = best_to_cp1(p0, v0, an, cp0, cp1)
        if pre_turn: pt, thrust = direction, force

        # 用距離判斷小狗
        if distance(p0,cp0) < 3000:  # 太遠的不用考慮小狗，此值來自實際發生的地球軌道之遠日點
            saved_context = p0,v0,an,pt,thrust  # save 準備做測試
            dogie_flag = True  # 先假設是「小狗兜圈子」的狀況
            for dogie in range(10):  # [ ] 可能要改更小更好？怎麼評估？
                pt, thrust = round(cp0-inertia(v0,p0,cp0)).astype(int), best_thrust(p0, v0, an, cp0)
                p0,v0,an,dist = move(p0,v0,an,(*pt,thrust),cp0)
                if dist <= cp_spot:
                    # 不是「小狗兜圈子」的狀況，放心了。
                    dogie_flag = False
                    break 
            p0,v0,an,pt,thrust = saved_context  # restore
            if dogie_flag: thrust = 0                    

        # Shield
        next_positions = ([(pod[i]['x']+pod[i]['vx'],pod[i]['y']+pod[i]['vy']) for i in (0,1)] 
                       + [(opp[i]['x']+opp[i]['vx'],opp[i]['y']+opp[i]['vy']) for i in (0,1)])
        # if i == 0:
        #     thrust = collision_detect_pod0(thrust,next_positions)
        # else:
        #     thrust = collision_detect_pod1(thrust,next_positions)
        thrust = collision_detect_pod0(thrust,next_positions)  # 都用同一套 shielding 策略

        thrust = thrust if type(thrust) == str else norm(thrust).astype(int)
        tx, ty = *pt,;
        print(f"{tx} {ty} {thrust}")
        if i == 0 : 
            pod0_drive = (tx, ty, thrust)
        else:
            pod1_drive = (tx, ty, thrust)

    ## 「領先-跟隨」策略
    #for i in (1,):
    #    # read data
    #    x, y, vx, vy, an, idx = pod[i].values();
    #    nnx, nny = cp[0] if (idx + 1) >= len(cp) else cp[idx+1]  # next next checkpoint
    #
    #    p0 = v(x,y)
    #    v0 = v(vx,vy)
    #    
    #    # 「領先-跟隨」或「跟隨-攔截」
    #    # 如果 achievements 相同，距離 np 比 striker 近的也當成 leader
    #    achievements = list(map(len, pod_history + opp_history))  # achievement of cp counts
    #    leader = np.argmax(achievements) # (0 1 2 3 : pod0 pod1 opp0 opp1)
    #    opp_leader = np.argmax(achievements[2:]) # (0 1 : opp0 opp1)
    #    # 已知 opp_leader 把要用到的東西都準備好，不管有沒有用。
    #    opp_p0 = v(opp[opp_leader]['x'], opp[opp_leader]['y'])
    #    opp_v0 = v(opp[opp_leader]['vx'],opp[opp_leader]['vy'])
    #    opp_np_idx = opp[opp_leader]['id']
    #    opp_nnp_idx = 0 if (opp_np_idx + 1) >= len(cp) else opp_np_idx + 1
    #    opp_np = v(*cp[opp_np_idx])
    #    opp_nnp = v(*cp[opp_nnp_idx])
    #    opp_2_np_distance = distance(opp_p0, opp_np)
    #
    #    # 誰接近 np?
    #    if opp_2_np_distance < distance(p0, opp_np) :
    #        # 追不上了，改前往 opp 的 np-nnp 中途等它。
    #        pt = opp_np + uv(opp_nnp - opp_np) * (distance(opp_nnp,opp_np)/2)  # 攔截點位
    #    else:
    #        # 駛向 opp-striker 到 np 的「半途」。
    #        pt = opp_p0 + uv(opp_np - opp_p0 ) * (opp_2_np_distance/2)
    #    pt = pt - inertia(v0,p0,pt)  # 修正慣性衝力
    #    # 有了目標就可以算出 thrust 了
    #    thrust = best_thrust(p0,v0,an,pt)
    #    # 越接用滑的，避免衝過頭。
    #    if enough_to_spot(p0, v0, pt, 17, tolerance=100): # 22 不到位改 17看看會不會過頭
    #        thrust = 0
    #
    #
    #    # Shield
    #    next_positions = ([(pod[i]['x']+pod[i]['vx'],pod[i]['y']+pod[i]['vy']) for i in (0,1)] 
    #                   + [(opp[i]['x']+opp[i]['vx'],opp[i]['y']+opp[i]['vy']) for i in (0,1)])
    #    if i == 0:
    #        thrust = collision_detect_pod0(thrust,next_positions)
    #    else:
    #        thrust = collision_detect_pod1(thrust,next_positions)
    #
    #    thrust = thrust if type(thrust) == str else norm(thrust).astype(int)
    #    tx,ty = pt.astype(int)
    #    print(f"{tx} {ty} {thrust}")
    #    pod1_drive = (tx, ty, thrust)

    #    # Blocker 需要紀錄與其他三人的前一距離，判斷是否接近中。 pod0,opp0,opp1
    #    blocker_last_distance[0] = distance(v(pod[1]['x'], pod[1]['y']), v(pod[0]['x'], pod[0]['y']))
    #    blocker_last_distance[1] = distance(v(pod[1]['x'], pod[1]['y']), v(opp[0]['x'], opp[0]['y']))
    #    blocker_last_distance[2] = distance(v(pod[1]['x'], pod[1]['y']), v(opp[1]['x'], opp[1]['y']))

    print(f"-----------------------------------------------------------", file=sys.stderr)
    print(f"cp={cp}", file=sys.stderr)
    print(f"'Step {step} pod0'; (x,y,vx,vy,an,idx)={[i for i in pod[0].values()]}  # Thrust={pod0_drive}", file=sys.stderr)
    print(f"'Step {step} pod1'; (x,y,vx,vy,an,idx)={[i for i in pod[1].values()]}  # Thrust={pod1_drive}", file=sys.stderr)
    print(f"'Step {step} opp0'; (x,y,vx,vy,an,idx)={[i for i in opp[0].values()]}", file=sys.stderr)
    print(f"'Step {step} opp1'; (x,y,vx,vy,an,idx)={[i for i in opp[1].values()]}", file=sys.stderr)
    print(f"pod_history {pod_history}", file=sys.stderr)
    print(f"opp_history {opp_history}", file=sys.stderr)
    print(f"pre_turn {pre_turn}", file=sys.stderr)
    print(f"-----------------------------------------------------------", file=sys.stderr)
    
    step += 1

