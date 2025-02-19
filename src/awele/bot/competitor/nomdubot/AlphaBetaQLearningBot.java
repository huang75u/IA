package awele.bot.competitor.nomdubot;

import awele.bot.CompetitorBot;
import awele.core.Board;
import awele.core.InvalidBotException;
import awele.data.AweleData;
import awele.data.AweleObservation;

import java.lang.reflect.Field;
import java.util.*;

/**
 * Bot (Q-Learning offline + Alpha-Beta depth=3 + TT + Move Ordering + PV/Killer + Evalu avancée)
 * - Apprentissage Q-Learning sur 303 obs, avec captureReward offline.
 * - getDecision : iterative deepening (1..MAX_DEPTH=3), PV & killer moves en move ordering.
 * - evaluate(...) : vantage + LAMBDA*q + potGrabs - oppPotGrabs - oppNextCapture, endgame multiplier.
 */
public class AlphaBetaQLearningBot extends CompetitorBot {
    //==================== Paramètres Principaux ====================

    /** Profondeur = 3，与 MinMaxBot(3)相同 */
    private static final int MAX_DEPTH = 3;

    /** Q-learning offline */
    private static final int NB_EPOCH = 200;    //迭代次数
    private static final double ALPHA_INIT = 0.1;
    private static final double ALPHA_DECAY = 0.995;
    private static final double GAMMA = 0.9;

    private static final double REWARD_WIN = 1.0;
    private static final double REWARD_LOSE = -1.0;
    /** 离线对可抓子额外奖励系数 */
    private static final double CAPTURE_REWARD_FACTOR_OFFLINE = 0.2;

    //==================== Évaluation / MoveOrdering ====================

    /** Bonus immédiat de capture (move ordering) */
    private static final double CAPTURE_BONUS = 3.5;

    /** Poids du Q值 */
    private static final double LAMBDA = 2.2;

    /** Endgame si nbSeeds <=12 => vantage *= 1.8 */
    private static final int ENDGAME_THRESHOLD = 12;
    private static final double ENDGAME_MULT = 1.8;

    /** Pénalise adversaire下一步可抓子 */
    private static final double OPP_CAPTURE_PENALTY = 1.7;

    /** 对自己洞(1或2粒)的潜在连抓加分 */
    private static final double MY_POTENTIAL_GRAB_BONUS = 0.4;

    /** 对对手洞(1或2粒)的潜在连抓的负面 */
    private static final double OPP_POTENTIAL_GRAB_PENALTY = 0.4;

    /** Encodage base (Board 12 trous) */
    private static final int BASE_ENCODE = 30;

    //==================== Données internes ====================

    private Map<Long, double[]> qTable;        // Q表： state-> Q[]
    private Map<TTKey, TTEntry> transposition; // table de transposition

    private double alpha;
    private int rootPlayer;

    // Principal Variation & Killer Moves pour alphaBeta
    private int[] principalMove;
    private int[][] killerMoves;

    //==================== Constructeur ====================

    public AlphaBetaQLearningBot() throws InvalidBotException {
        this.setBotName("AlphaBetaQL_Enhanced");
        this.addAuthor("Auteur1");
        this.addAuthor("Auteur2");

        this.qTable = new HashMap<>();
        this.transposition = new HashMap<>();
        this.alpha = ALPHA_INIT;

        this.principalMove = new int[MAX_DEPTH+1];
        Arrays.fill(this.principalMove, -1);

        this.killerMoves = new int[MAX_DEPTH+1][2];
        for(int d=0; d<=MAX_DEPTH; d++){
            Arrays.fill(this.killerMoves[d], -1);
        }
    }

    //==================== Q-Learning Offline ====================

    @Override
    public void learn() {
        AweleData data = AweleData.getInstance();
        AweleObservation[] obsArr = data.toArray(new AweleObservation[0]);

        for(int epoch=0; epoch<NB_EPOCH; epoch++){
            for(AweleObservation obs: obsArr){
                long sKey = encodeObservation(obs);
                int action = obs.getMove() - 1;

                // base reward : gagné(+1)/perdu(-1)
                double r = obs.isWon()? REWARD_WIN : REWARD_LOSE;

                // On essaie de reconstruire le Board, pour calculer combien de graines peut-capturer offline.
                double captureSeeds = 0;
                try {
                    Board b = reconstructBoard(obs);
                    int sc = b.playMoveSimulationScore(b.getCurrentPlayer(), moveVector(action));
                    if(sc>0) {
                        captureSeeds = sc;
                    }
                } catch(Exception e) {
                    // e.printStackTrace(); //ou ignorer
                }
                r += CAPTURE_REWARD_FACTOR_OFFLINE * captureSeeds;

                double[] qVals = this.qTable.getOrDefault(sKey, new double[Board.NB_HOLES]);
                double oldQ = qVals[action];
                // nextState non déterminé => nextQ=0 simplifié
                double target = r + GAMMA*0;
                qVals[action] = oldQ + alpha*(target - oldQ);
                this.qTable.put(sKey, qVals);
            }
            alpha *= ALPHA_DECAY;
        }
        System.out.println("[AlphaBetaQL_Enhanced] Q-table size= " + qTable.size());
    }

    /** Reconstituer Board via reflection */
    private Board reconstructBoard(AweleObservation obs) throws Exception {
        Board board = new Board();
        // set holes by reflection
        Field holesF = Board.class.getDeclaredField("holes");
        holesF.setAccessible(true);
        int[][] holes = new int[2][6];

        // Suppose on fait obs.getPlayerHoles() -> holes[0], obs.getOppenentHoles()-> holes[1], currentPlayer=0.
        System.arraycopy(obs.getPlayerHoles(), 0, holes[0], 0, 6);
        System.arraycopy(obs.getOppenentHoles(), 0, holes[1], 0, 6);

        holesF.set(board, holes);

        Field curF = Board.class.getDeclaredField("currentPlayer");
        curF.setAccessible(true);
        curF.setInt(board, 0);

        return board;
    }

    @Override
    public void initialize() {
        this.transposition.clear();
        Arrays.fill(this.principalMove, -1);
        for(int d=0; d<=MAX_DEPTH; d++){
            Arrays.fill(this.killerMoves[d], -1);
        }
    }
    @Override
    public void finish() {}

    //==================== getDecision => iterative deepening ====================

    @Override
    public double[] getDecision(Board board) {
        this.rootPlayer = board.getCurrentPlayer();
        double[] bestMoves = null;

        for(int depth=1; depth<=MAX_DEPTH; depth++){
            double[] decisionDepth = new double[Board.NB_HOLES];
            MoveInfo[] moveInfos = new MoveInfo[Board.NB_HOLES];

            // calcul heur, puis bonus PV/killer
            for(int m=0; m<Board.NB_HOLES; m++){
                try{
                    int sc = board.playMoveSimulationScore(rootPlayer, moveVector(m));
                    if(sc<0){
                        moveInfos[m]= new MoveInfo(m, Double.NEGATIVE_INFINITY);
                    } else {
                        Board nb = board.playMoveSimulationBoard(rootPlayer, moveVector(m));
                        double cBonus= sc * CAPTURE_BONUS;
                        double qv= getMaxQValue(nb);
                        double oppCap= simulateOpponentCapture(nb, 1-rootPlayer);

                        double heur= cBonus + LAMBDA*qv - oppCap;
                        moveInfos[m]= new MoveInfo(m, heur);
                    }
                } catch(InvalidBotException e){
                    moveInfos[m]= new MoveInfo(m,Double.NEGATIVE_INFINITY);
                }
            }
            // principalMove bonus
            int pvM= this.principalMove[depth];
            if(pvM>=0 && pvM<Board.NB_HOLES && moveInfos[pvM].heuristic!=Double.NEGATIVE_INFINITY){
                moveInfos[pvM].heuristic += 100_000;
            }
            // killerMoves bonus
            for(int km=0; km<2; km++){
                int killM= this.killerMoves[depth][km];
                if(killM>=0 && killM<Board.NB_HOLES && moveInfos[killM].heuristic!=Double.NEGATIVE_INFINITY){
                    moveInfos[killM].heuristic += 50_000;
                }
            }

            Arrays.sort(moveInfos, (a,b)->Double.compare(b.heuristic,a.heuristic));

            double bestVal= Double.NEGATIVE_INFINITY;
            int bestIdx=-1;

            for(MoveInfo mi: moveInfos){
                if(mi.heuristic==Double.NEGATIVE_INFINITY){
                    decisionDepth[mi.move]= Double.NEGATIVE_INFINITY;
                } else {
                    try{
                        Board child= board.playMoveSimulationBoard(rootPlayer, moveVector(mi.move));
                        double val= alphaBeta(child,1,depth,Double.NEGATIVE_INFINITY,Double.POSITIVE_INFINITY,false);
                        decisionDepth[mi.move]= val;
                        if(val>bestVal){
                            bestVal= val;
                            bestIdx= mi.move;
                        }
                    }catch(InvalidBotException e){
                        decisionDepth[mi.move]=Double.NEGATIVE_INFINITY;
                    }
                }
            }
            if(bestIdx>=0) this.principalMove[depth]= bestIdx;
            bestMoves= decisionDepth;
        }
        return bestMoves;
    }

    //==================== alphaBeta with TT + killer ====================

    private double alphaBeta(Board board, int currentDepth, int maxDepth, double alpha, double beta, boolean isMax){
        if(currentDepth>=maxDepth || isTerminal(board)){
            return evaluate(board);
        }
        TTKey key= new TTKey(encodeBoard(board), maxDepth-currentDepth, isMax, board.getCurrentPlayer());
        TTEntry entry= transposition.get(key);
        if(entry!=null){
            if(entry.lowerBound>= beta) return entry.value;
            if(entry.upperBound<= alpha) return entry.value;
            alpha= Math.max(alpha, entry.lowerBound);
            beta= Math.min(beta, entry.upperBound);
        }

        double bestVal= isMax? Double.NEGATIVE_INFINITY: Double.POSITIVE_INFINITY;
        MoveInfo[] moveInfos= new MoveInfo[Board.NB_HOLES];
        for(int m=0; m<Board.NB_HOLES; m++){
            try{
                int sc= board.playMoveSimulationScore(board.getCurrentPlayer(), moveVector(m));
                if(sc<0){
                    moveInfos[m]= new MoveInfo(m, Double.NEGATIVE_INFINITY);
                } else {
                    Board nb= board.playMoveSimulationBoard(board.getCurrentPlayer(), moveVector(m));
                    double cBonus= sc* CAPTURE_BONUS;
                    double qv= getMaxQValue(nb);
                    double oppCap= simulateOpponentCapture(nb, 1-board.getCurrentPlayer());
                    double heur= cBonus + LAMBDA*qv - oppCap;
                    moveInfos[m]= new MoveInfo(m, heur);
                }
            } catch(InvalidBotException e){
                moveInfos[m]= new MoveInfo(m,Double.NEGATIVE_INFINITY);
            }
        }
        Arrays.sort(moveInfos, (a,b)->Double.compare(b.heuristic,a.heuristic));

        boolean cutoff=false;
        int cutoffMove=-1;
        if(isMax){
            for(MoveInfo mi: moveInfos){
                if(mi.heuristic==Double.NEGATIVE_INFINITY) continue;
                try{
                    Board child= board.playMoveSimulationBoard(board.getCurrentPlayer(), moveVector(mi.move));
                    double val= alphaBeta(child, currentDepth+1, maxDepth, alpha,beta,false);
                    if(val>bestVal){
                        bestVal= val;
                        cutoffMove= mi.move;
                    }
                    alpha= Math.max(alpha, bestVal);
                    if(alpha>=beta){
                        cutoff=true;
                        break;
                    }
                } catch(InvalidBotException ignored){}
            }
        } else {
            for(MoveInfo mi: moveInfos){
                if(mi.heuristic==Double.NEGATIVE_INFINITY) continue;
                try{
                    Board child= board.playMoveSimulationBoard(board.getCurrentPlayer(), moveVector(mi.move));
                    double val= alphaBeta(child, currentDepth+1, maxDepth, alpha,beta,true);
                    if(val<bestVal){
                        bestVal= val;
                        cutoffMove= mi.move;
                    }
                    beta= Math.min(beta, bestVal);
                    if(alpha>=beta){
                        cutoff=true;
                        break;
                    }
                } catch(InvalidBotException ignored){}
            }
        }

        // killer moves
        if(cutoff && cutoffMove>=0){
            int[] kms= killerMoves[currentDepth];
            if(kms[0]!= cutoffMove){
                kms[1]= kms[0];
                kms[0]= cutoffMove;
            }
        }

        TTEntry newEntry= new TTEntry();
        newEntry.value= bestVal;
        if(bestVal<= alpha){
            newEntry.upperBound= bestVal;
            newEntry.lowerBound= Double.NEGATIVE_INFINITY;
        } else if(bestVal>= beta){
            newEntry.lowerBound= bestVal;
            newEntry.upperBound= Double.POSITIVE_INFINITY;
        } else {
            newEntry.lowerBound= bestVal;
            newEntry.upperBound= bestVal;
        }
        transposition.put(key,newEntry);

        return bestVal;
    }

    //==================== Évaluation ====================

    private boolean isTerminal(Board b){
        return b.getScore(0)>=25 || b.getScore(1)>=25 || b.getNbSeeds()<=6;
    }

    /**
     * vantage + LAMBDA*q - oppCap*OPP_CAPTURE_PENALTY + myPotential - oppPotential, endgame多倍
     */
    private double evaluate(Board b){
        double vantage= b.getScore(rootPlayer)- b.getScore(1-rootPlayer);
        if(b.getNbSeeds()<=ENDGAME_THRESHOLD){
            vantage*= ENDGAME_MULT;
        }
        double qv= getMaxQValue(b);
        double oppCap= simulateOpponentCapture(b, 1-rootPlayer);
        double penalty= oppCap* OPP_CAPTURE_PENALTY;

        double myPot= evaluateMyPotential(b);
        double oppPot= evaluateOppPotential(b);

        return vantage + LAMBDA*qv - penalty + myPot - oppPot;
    }

    /**
     * 统计 rootPlayer的洞(1或2粒)数量 * MY_POTENTIAL_GRAB_BONUS
     */
    private double evaluateMyPotential(Board b){
        int[] myHoles = getRootPlayerHoles(b);
        int count=0;
        for(int h: myHoles){
            if(h==1 || h==2) count++;
        }
        return count*MY_POTENTIAL_GRAB_BONUS;
    }

    /**
     * 统计 对手(1-rootPlayer)的洞(1或2粒)数量 * OPP_POTENTIAL_GRAB_PENALTY
     */
    private double evaluateOppPotential(Board b){
        int[] oppHoles = getOpponentOfRootHoles(b);
        int count=0;
        for(int h: oppHoles){
            if(h==1 || h==2) count++;
        }
        return count*OPP_POTENTIAL_GRAB_PENALTY;
    }

    /**
     * rootPlayer的6个洞
     */
    private int[] getRootPlayerHoles(Board b){
        // 若 b.getCurrentPlayer()==rootPlayer => b.getPlayerHoles() 就是 rootPlayer 的
        // 否则 b.getOpponentHoles() 是 rootPlayer 的
        if(b.getCurrentPlayer() == rootPlayer){
            return b.getPlayerHoles();
        } else {
            return b.getOpponentHoles();
        }
    }

    /**
     * 对手(1-rootPlayer)的6个洞
     */
    private int[] getOpponentOfRootHoles(Board b){
        int opp = 1-rootPlayer;
        if(b.getCurrentPlayer() == opp){
            return b.getPlayerHoles();
        } else {
            return b.getOpponentHoles();
        }
    }

    private double getMaxQValue(Board b){
        long code= encodeBoard(b);
        double[] arr= qTable.getOrDefault(code,new double[Board.NB_HOLES]);
        double mx= Double.NEGATIVE_INFINITY;
        for(double v: arr){
            if(v>mx) mx=v;
        }
        return (mx<0)? 0: mx; // clamp <0 =>0
    }

    private double simulateOpponentCapture(Board b,int opp){
        int maxCap=0;
        for(int m=0;m<Board.NB_HOLES;m++){
            try{
                int sc= b.playMoveSimulationScore(opp, moveVector(m));
                if(sc>maxCap) maxCap=sc;
            } catch(InvalidBotException ignored){}
        }
        return maxCap;
    }

    private double[] moveVector(int move){
        double[] arr = new double[Board.NB_HOLES];
        arr[move]=1.0;
        return arr;
    }

    //==================== Encodage (Q-table) ====================

    private long encodeObservation(AweleObservation obs){
        int[] p= obs.getPlayerHoles();
        int[] o= obs.getOppenentHoles();
        return encodeHoles(p,o);
    }

    private long encodeBoard(Board b){
        return encodeHoles(b.getPlayerHoles(), b.getOpponentHoles());
    }

    private long encodeHoles(int[] ph, int[] oh){
        long code=0;
        for(int x: ph) code= code*BASE_ENCODE + x;
        for(int x: oh) code= code*BASE_ENCODE + x;
        return code;
    }

    //==================== TTKey / TTEntry ====================

    private static class TTKey{
        long stateCode;
        int depth;
        boolean isMax;
        int currentPlayer;
        TTKey(long s,int d, boolean m, int cp){
            stateCode=s; depth=d; isMax=m; currentPlayer=cp;
        }
        @Override
        public boolean equals(Object o){
            if(!(o instanceof TTKey)) return false;
            TTKey k=(TTKey)o;
            return (stateCode==k.stateCode && depth==k.depth && isMax==k.isMax && currentPlayer==k.currentPlayer);
        }
        @Override
        public int hashCode(){
            return Objects.hash(stateCode, depth, isMax, currentPlayer);
        }
    }
    private static class TTEntry{
        double value;
        double lowerBound=Double.NEGATIVE_INFINITY;
        double upperBound=Double.POSITIVE_INFINITY;
    }

    //==================== MoveInfo / killer ====================

    private static class MoveInfo{
        int move;
        double heuristic;
        MoveInfo(int m,double h){ move=m; heuristic=h;}
    }
}
