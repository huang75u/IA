package awele.bot.competitor.nomdubot;

import awele.bot.CompetitorBot;
import awele.core.Board;
import awele.core.InvalidBotException;
import awele.data.AweleData;
import awele.data.AweleObservation;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * 示例：一个基于Q-Learning的Awélé对手bot。
 *  - 使用离线训练：在 learn() 中读 303 条数据并更新 Q表
 *  - 在 getDecision() 中用 Q表做决策
 */
public class QLearningBot extends CompetitorBot {
    /**
     * Q表：key是对局面进行哈希后的Long值，value是长度=6的double[]，表示对6个坑位的Q(s,a)
     */
    private Map<Long, double[]> qTable;

    /** 学习率 alpha */
    private static final double ALPHA = 0.1;

    /** 折扣因子 gamma（本示例只用离线数据，故可设小些或0） */
    private static final double GAMMA = 0.0;

    /**
     * 为了演示，本例将一个观测视为 (S, A, R, S')，其中 R = +1 (若观测来自赢方)，-1 (若输方)。
     * 然后离线多次扫数据更新 Q 值。
     */
    private static final double REWARD_WIN = 1.0;
    private static final double REWARD_LOSE = -1.0;

    /**
     * 训练迭代次数(离线)：可增大以强化Q值收敛
     * 303条观测 * NB_EPOCH 次
     */
    private static final int NB_EPOCH = 50;

    /** 随机数，用于 epsilon贪心(如果想的话) */
    private Random random;

    /**
     * 构造函数(无参)，添加作者与Bot名
     */
    public QLearningBot() throws InvalidBotException {
        this.addAuthor("张三");
        this.addAuthor("李四");
        this.setBotName("QlearningBot_Demo");

        this.qTable = new HashMap<>();
        this.random = new Random(System.currentTimeMillis());
    }

    /**
     * Offline 学习阶段(在Bot加载时只执行一次)，
     * 从 AweleData 读取 303 条观测并进行若干次迭代(离线Q-Learning)。
     */
    @Override
    public void learn() {
        AweleData data = AweleData.getInstance();
        // 先把观测读到内存
        AweleObservation[] observations = data.toArray(new AweleObservation[0]);

        // 简单离线训练：多次重复遍历Observations
        for(int epoch=0; epoch<NB_EPOCH; epoch++){
            for (AweleObservation obs : observations) {
                // 构建 state
                long stateKey = encodeObservation(obs);
                // 取 move (1..6)，转成下标 (0..5)
                int action = obs.getMove() - 1;
                // reward
                double r = obs.isWon()? REWARD_WIN : REWARD_LOSE;

                // oldQ
                double[] qValues = this.qTable.getOrDefault(stateKey, new double[Board.NB_HOLES]);
                // Q(s,a) <- Q(s,a) + alpha * [ r + gamma*max Q(s',.) - Q(s,a)]
                // 这里因为我们不提供 s' (下一状态) 也无后继观测，所以 gamma=0
                double oldVal = qValues[action];
                double newVal = oldVal + ALPHA*(r - oldVal); // gamma=0 => (r + 0 - oldVal)
                qValues[action] = newVal;

                // 存回Q表
                this.qTable.put(stateKey, qValues);
            }
        }
        System.out.println("Q-Learning offline finished. Q-table size="+ qTable.size());
    }

    /**
     * 每盘对局开始前调用
     */
    @Override
    public void initialize() {
        // 可以在这里做一些重置操作(若需要)
    }

    /**
     * 核心决策：根据当前board，返回6个槽位的价值。
     *  - 先把Board编码成Long key
     *  - 若Q表中有则用其值；没有则默认0(或随机)
     */
    @Override
    public double[] getDecision(Board board) {
        double[] decision = new double[Board.NB_HOLES];

        // 编码状态
        long stateKey = encodeBoard(board);

        // 查找Q(s,.)
        double[] qValues = this.qTable.getOrDefault(stateKey, null);
        if(qValues == null){
            // 若没出现过这个状态(或仅在数据外)，则全部给0.0
            // 也可以小扰动 random
            for(int i=0; i<Board.NB_HOLES; i++){
                decision[i] = 0.0;
            }
        } else {
            // 把 Q(s,a) 直接当做决策值
            System.arraycopy(qValues, 0, decision, 0, Board.NB_HOLES);
        }

        return decision;
    }

    /**
     * 对局结束后调用
     */
    @Override
    public void finish() {
        // 可做一些统计或资源释放
    }

    // --------------------------------------------------------------
    // 下面是辅助方法：状态编码 / 从观察构造状态
    // --------------------------------------------------------------

    /**
     * 将 AweleObservation (12个坑分布 + 当前玩家) 转成long，用于哈希key
     * @param obs AweleObservation
     */
    private long encodeObservation(AweleObservation obs){
        // PDF说明 12 个变量： [A6..A1(玩家侧), J1..J6(对手侧)] + (实际上观测中并未独立给当前玩家ID，但题意表明 "第13列是move, 第14列是G/P" )
        // 这里假设玩家固定就是 “下方6坑” -> obs.getPlayerHoles() => [A1..A6]
        // 对手 -> obs.getOppenentHoles() => [A1..A6], 但顺序在题目描述是反向
        // 为简单起见，咱们直接把obs.getPlayerHoles() + obs.getOppenentHoles()按固定顺序拼起来
        // （如果你想严格跟 Board 的 "currentPlayer" 一致，可进一步区分是谁先手，但本示例就简化处理）
        int[] p = obs.getPlayerHoles();   // length=6
        int[] o = obs.getOppenentHoles(); // length=6
        long code = 0;
        // 让每个坑的种子数量(0..??)在某个进制中叠加
        // 这里简单假设每坑最多20粒(随便取个上限)
        // code = sum( p[i]* (21^i) ) + sum( o[i]* (21^(i+6)) ) ...
        // 21是因为0..20共21个可能值
        int base = 21;
        for(int i=0; i<6; i++){
            code = code*base + p[i];
        }
        for(int i=0; i<6; i++){
            code = code*base + o[i];
        }
        // 这样就得到一个long值
        return code;
    }

    /**
     * 将当前Board转换成long key，用于在Q表中查询
     */
    private long encodeBoard(Board board){
        // 类似上面 encodeObservation 的做法，只不过Board里我们能直接
        //   board.getPlayerHoles() / board.getOpponentHoles() 拿到数组
        //   并用同样21进制叠加
        int[] p = board.getPlayerHoles();   // 6坑
        int[] o = board.getOpponentHoles(); // 6坑
        long code = 0;
        int base = 21;
        for(int i=0; i<6; i++){
            code = code*base + p[i];
        }
        for(int i=0; i<6; i++){
            code = code*base + o[i];
        }
        return code;
    }
}
