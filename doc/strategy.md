# 2021智荟杯 捕碳大作战 规则策略

## Halite 策略借鉴

- Halite和本次的比赛规则很类似。
- Halite分为4个版本，即I,II,III,IV
- 其中Halite III的规则和捕碳最接近
- IV的规则与捕碳的主要区别在于(我简要概述一下)：1.没有种树员 2.可以用一定花费使捕碳员(Halite中叫ship/船)变为转化中心(Halite中叫shipyard) 3. 是四人游戏，而不是两人 4.可以用ship攻击shipyard。以下是对Halite IV方案的借鉴，但因为规则不同，需要适当改编

### Halite核心规则

- Halite is a four-player game taking place on a 21x21 grid map over 400 turns in which the players choose their actions simultaneously.
- Each player starts with 5000 halite (the central resource in the game) and one ship
- Ships can choose from six actions: they can move in four directions (north, east, south and west), stand still or convert into a shipyard for 500 halite
- A shipyard can spawn one ship per turn for a fee of 500 halite.
- Halite is placed symmetrically on many of the cells of the game map.
- When a ship stays still on a cell with halite, it mines 25% of the cell's halite each turn receiving this amount as cargo.
- Each turn halite cells with no ships on them regenerate 2% of their halite value up to a maximum of 500 halite.
- Ships must return their cargo by moving on top of a shipyard to get their cargo transferred to the bank of the player from which the halite can be spent.
- When multiple ships move to the same cell at the same turn, the ship with the least cargo survives and receives the cargo of the other ships which get destroyed. All ships get destroyed in case of a tie.
- Ships crashing into enemy shipyards destroy themselves and the shipyard if the shipyard doesn't have a ship on top of it or spawns one.
- When a player has no ships and shipyards left (or only shipyards and not enough halite to spawn a ship), he is immediately eliminated from the game.
- At the end of the game the surviving players are ranked based on the halite in their bank.



### **Halite IV 1st**

https://www.kaggle.com/c/halite/discussion/183543

#### **代码**

https://storage.googleapis.com/kaggle-forum-message-attachments/1013928/17023/Rule%20actions%20v3%20optimum%201%20additional%20rules%2029%20-%202.py

#### **博客内容**

用规则，不用RL

单位之间的合作是关键，不打算为每个单位独立地设计策略

策略=3个步骤+特殊策略

##### **3个步骤**

计算agent-action score; 计算ship plan; factor ship plan to actions

##### **特殊策略**

###### **Camping strategy（猥琐发育）**

如果比较近的地方有对手攻击，就围在基地旁边转圈

###### **Base construction**

每9个捕碳者建一个基地(在捕碳中无效，因为只有一个基地)

###### **Base defence**

无效 所以不写了

###### **Opponent ship boxing in**

重点是要预测对手捕碳者的行动，逃跑路线以确保击杀

###### **Base attack**

无效 所以不写了

###### **Opponent hunting**

无效 所以不写了

因为我们是1v1零和游戏

###### **Rescue missions**

##### **Curated list of low level features**

让低碳(战斗力强)的捕碳者营救没位置逃的或者正在被追的同伴

###### **Opponent zero halite model**

检测对手0碳船的位置来预测风险等级

###### **Opponent conversion model**

无效 所以不写了

###### **1-step, 2-step and N-step bad actions**

在计算agent-action score时专门找出bad action不去选它

- 1 step: if I take this action, I can lose the ship if the opponent takes some action (this is the worst scenario)
- 2 step: under optimal opponent play, my ship would have no valid escape path in the next step, if I take that action
- N step: If I take this action, I may be surrounded by opponents in all directions or have no safe path home. In this scenario, I compute a risk score as a function of the number and distance of threats and mark an action as bad if it exceeds some dynamic threshold (function of game step and number of lost ships). If there are several N-step bad actions: limit the choice to the least bad ones.

###### **Chase detection**

combination with rescue missions

###### **Cycle detection**

如果单位堵在一起了，就把单位连在一起规划下一步的行动

###### **Safe collect squares**

###### **Initial halite collect override near base**

游戏初期，不要在基地旁边捕碳，而是从对手的活动边界上捕碳

###### **Avoid attack squares**

不在地图的拥挤部分攻击对手可能的逃生方格。我不理解。

###### **Escape square selection**

当己方单位和地方单位隔得很近（近身）时，需要在上述3个策略的factor ship plan to actions阶段进行精细分析，比如分析对手的单次移动、搜索逃生路线

###### **Zero halite boxed in ship**

0碳船可能 为了保护友方单位 而和对方同归于尽

### **Halite IV 2nd**

https://www.kaggle.com/c/halite/discussion/186032



### Halite IV 4th

https://www.kaggle.com/c/halite/discussion/183727

将 shipyards 建在 halite 资源密集区

等待 halite 生长到一定阈值后进行开采

在保护 shipyard ，吓跑敌人(通过空ship)的同时，希望一些敌人在我们的开采区开采(诱敌深入)来消灭敌人，对敌人造成损失

不记录之前的状态，每次的决策基于当前地图情况，以减少错误决策的影响。只记录很少的全局信息：shipyard 的位置和数量等。

船只类型：

1. 采矿船：前往 hatile 密集区进行采矿
2. 返航船：拥有较多的 halite 资源，准备前往 shipyard 卸货
3. 捕猎船：拥有 0 hatile，试图去撞击敌船
4. 保卫船：拥有 0 hatile，在边界巡逻以保护 shipyard
5. 防御船：在自己区域里的捕猎船

打分函数：

使用贪心的策略，采矿船取矿最多的地方，捕猎船取击杀采矿最多的敌船，防御船待在敌人最多的地方

\+ 捕猎成绩最差的的捕猎船变成防御船

\+ 试图让两艘防御船与造船厂的距离和与造船厂的距离最近的两艘敌船的距离相等

\+ 每当船舶数量超过一定的阈值时，开始计算下一个造船厂的位置。在规划了一个新的造船厂后，增加在选定造船厂附近的敌人目标的狩猎分数，慢慢地巩固了阵地。然后我们派出两艘船只，一艘护卫舰确保新建造的造船厂不会立即被摧毁，另一艘建造舰到达该位置后将转变为造船厂。

建造shipyard策略：

1 我们将第一个造船厂放置在富含石盐的电池组附近，以便在游戏中有一个良好的开端。

2 第二个造船厂也位于富含石盐的区域，但我们也考虑了两个造船厂之间区域内的石盐数量。

3 我们放置第三个造船厂，以使包含的区域最大限度地增加石盐数量，但我们也尽量不将其放置在敌方造船厂附近，因为敌方造船厂的玩家并未完全退出游戏。

4 在第三个造船厂之后，我们尝试最大化岩盐数量和优势值（使用非常模糊的优势图）。

区域图

​                 ![img](https://docimg5.docs.qq.com/image/xkMAWVTSxGTds7H_x7FkwA.png?w=500&h=380)        

1 农耕区：这些位置是我们用来收获石盐的区域

2 次要农耕区：这些也是农耕区，但它们不像真正的农耕区那样容易保护。因此，我们不让它们长得和其他的一样，也不惩罚在它们上面航行的船只

3 守卫区：这些位置上敌人的狩猎分数增加，当敌舰在那里停留太久时，我们的机器人会进行1:1的追杀。

4 边界位置：这仅仅是我们种植园周围的一个边界，我们的护卫舰沿着这个边界巡逻。

### **Halite IV 5th**

https://www.kaggle.com/c/halite/discussion/183704

### [Halite IV 8th](https://www.kaggle.com/c/halite/discussion/183312)

- 如何在每轮移动捕碳员来捕捉更多的碳？
- 如何考虑智能体应该去哪些区域？比如：避免那些敌人密集的区域。
- 何时让捕碳员回到转化中心？比如：捕碳员携带太多的碳容易被攻击，但同时来回太频繁又影响捕碳的效率。
- 如何检测到落单的敌方捕碳员，同时如何移动我方捕碳员过去干掉他？
- 何时招募新的捕碳员和种树员？招募是一个重要因素，涉及猥琐发育策略还是以攻击为主的策略，前者（猥琐发育）就是少招人，多捕碳，后者（攻击）就是多招人，去攻击对手的捕碳员，或者抢树。
- 何时何地种树？
- 怎样追踪和把敌人排序，来选择最弱小/最容易攻击的敌人？
- ...

## **相关链接**

  游戏规则 介绍  https://github.com/moliqingwa/carbon_challenge 

  baseline 代码 https://github.com/buaa-054/carbon_baseline_cuda 

### **Halite**

#### **Halite III(我觉得是规则最接近捕碳的)**

https://2018.halite.io/

#### Halite IV Kaggle

优胜方案汇总 https://www.kaggle.com/c/halite/discussion/186290



  
