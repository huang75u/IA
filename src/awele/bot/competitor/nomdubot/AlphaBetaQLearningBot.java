package awele.bot.competitor.nomdubot;

import awele.bot.CompetitorBot;
import awele.core.Board;
import awele.core.InvalidBotException;
import awele.data.AweleData;
import awele.data.AweleObservation;

import java.lang.reflect.Field;
import java.util.*;

/**
 * Un Bot amélioré :
 * 1) Q-Learning avec récompenses partielles (ex. capture seed => bonus).
 * 2) Move ordering plus fin : prise en compte multi-facteurs (capture, seeds, Q, envoi seeds à l'adversaire).
 * 3) Transposition table plus complète : stocker alpha/beta + node type (LOWER,UPPER,EXACT).
 */
public class AlphaBetaQLearningBot extends CompetitorBot {
    //===================== Paramètres =================================

    private static final int MAX_DEPTH = 5;       // Profondeur max
    private static final int NB_EPOCH = 200;      // Itérations Q-learning
    private static final double ALPHA_INIT = 0.1; // Taux apprentissage initial
    private static final double ALPHA_DECAY = 0.995;
    private static final double GAMMA = 0.9;      // Facteur discount
    private static final double WIN_REWARD = 1.0;
    private static final double LOSE_REWARD = -1.0;
    private static final double CAPTURE_REWARD_FACTOR = 0.2; // Bonus par graine capturée offline

    // Évaluation
    private static final double CAPTURE_BONUS = 3.0;  // Move ordering bonus par graine
    private static final double LAMBDA = 2.0;         // Poids du Q
    private static final double SEEDS_WEIGHT = 0.5;   // Influence du total de graines
    private static final double SEND_RISK_PENALTY = 1.0; // Pénalité si on “envoie” trop
    private static final int BASE_ENCODE = 30;        // Encodage

    //===================== Structures =================================
    private Map<Long, double[]> qTable;  // Q(s)->[6 moves]
    private Map<TTKey, TTEntry> transposition; // TT

    private int rootPlayer;
    private double alpha; // Taux d'apprentissage adaptatif

    //===================== Constructeur =================================
    public AlphaBetaQLearningBot() throws InvalidBotException {
        this.setBotName("AlphaBetaQL_Advanced");
        this.addAuthor("YourName");

        this.qTable = new HashMap<>();
        this.transposition = new HashMap<>();
        this.alpha = ALPHA_INIT;
    }

    //===================== Q-Learning Offline ============================
    @Override
    public void learn() {
        AweleData data = AweleData.getInstance();
        AweleObservation[] obsArr = data.toArray(new AweleObservation[0]);

        for(int epoch=0; epoch<NB_EPOCH; epoch++){
            for(AweleObservation obs: obsArr){
                // Encodage de l'état
                long sKey = encodeObservation(obs);
                int a = obs.getMove()-1;

                // On suppose qu'on “estime” la capture offline comme 0 faute d'info
                // Si tu as la capture, tu peux faire:
                // double captureReward = CAPTURE_REWARD_FACTOR * [some captured seeds]
                double captureReward = 0;
                double r = (obs.isWon()? WIN_REWARD : LOSE_REWARD) + captureReward;

                // Q(s,a) update
                double[] qv = qTable.getOrDefault(sKey,new double[6]);
                double oldVal = qv[a];

                // On n'a pas S' exact => gamma=0 OU on tente un S' reconstitué
                // ex. Board nextBoard = simulateBoard(obs); ...
                // double maxNextQ = ...
                // On illustre seulement gamma=0 ou gamma*gmax=0
                double maxNextQ=0; // Simplify
                double target = r + GAMMA*maxNextQ;

                qv[a] = oldVal + alpha*(target - oldVal);
                qTable.put(sKey, qv);
            }
            alpha *= ALPHA_DECAY;
        }
        System.out.println("[AlphaBetaQL_Advanced] learn done. Q-table size="+qTable.size());
    }

    //===================== Structures simuler Board (si besoin) =====================
    // Si tu veux reconstituer un Board depuis un AweleObservation
    // pour calculer nextState etc., tu peux via reflect comme précédemment.
    /*
    private Board simulateBoard(AweleObservation obs){
        try {
            Board board = new Board();
            Field holesField = Board.class.getDeclaredField("holes");
            holesField.setAccessible(true);
            int[][] holes = new int[2][6];
            System.arraycopy(obs.getOppenentHoles(),0,holes[0],0,6);
            System.arraycopy(obs.getPlayerHoles(),0,holes[1],0,6);
            holesField.set(board,holes);

            Field currPlayerF = Board.class.getDeclaredField("currentPlayer");
            currPlayerF.setAccessible(true);
            currPlayerF.setInt(board,1);
            return board;
        } catch(Exception e){
            throw new RuntimeException(e);
        }
    }
    */

    //===================== Cycle de vie du Bot ==========================
    @Override
    public void initialize() {
        transposition.clear();
    }
    @Override
    public void finish() {}

    //===================== getDecision => IterativeDeepening + TT ==================
    @Override
    public double[] getDecision(Board board) {
        this.rootPlayer = board.getCurrentPlayer();

        double[] bestMoves = null;
        // Simple iterative deepening, on n'a pas le time limit code here
        for(int depth=1; depth<=MAX_DEPTH; depth++){
            double[] decisionAtDepth = new double[6];
            // On fait un move ordering global
            List<MoveInfo> moves = buildMoveInfos(board, rootPlayer);
            moves.sort(Comparator.comparingDouble(m->-m.heuristic)); // desc

            for(MoveInfo mi: moves){
                try{
                    if(mi.heuristic==Double.NEGATIVE_INFINITY){
                        decisionAtDepth[mi.move] = Double.NEGATIVE_INFINITY;
                    } else {
                        Board next = board.playMoveSimulationBoard(rootPlayer, moveVector(mi.move));
                        double val = alphaBeta(next, 1, depth,
                                Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, false);
                        decisionAtDepth[mi.move] = val;
                    }
                } catch(InvalidBotException e){
                    decisionAtDepth[mi.move]=Double.NEGATIVE_INFINITY;
                }
            }
            bestMoves = decisionAtDepth;
        }
        return bestMoves;
    }

    //===================== alphaBeta + TT plus complet ======================
    private double alphaBeta(Board board, int curDepth, int maxDepth,
                             double alpha, double beta, boolean isMax){
        if(curDepth>=maxDepth || isTerminal(board)) {
            return evaluate(board);
        }
        // TTKey with alpha/beta
        TTKey key = new TTKey(encodeBoard(board), maxDepth-curDepth, isMax, board.getCurrentPlayer());
        TTEntry entry = transposition.get(key);
        if(entry!=null){
            // check bounds
            if(entry.lowerBound >= beta) {
                return entry.value; // cutoff
            }
            if(entry.upperBound <= alpha){
                return entry.value; // cutoff
            }
            alpha = Math.max(alpha, entry.lowerBound);
            beta = Math.min(beta, entry.upperBound);
        }

        double bestVal;
        List<MoveInfo> moves = buildMoveInfos(board, board.getCurrentPlayer());
        moves.sort(Comparator.comparingDouble(m->-m.heuristic)); // desc

        if(isMax){
            bestVal = Double.NEGATIVE_INFINITY;
            for(MoveInfo mi: moves){
                if(mi.heuristic == Double.NEGATIVE_INFINITY) continue;
                try {
                    Board child = board.playMoveSimulationBoard(board.getCurrentPlayer(), moveVector(mi.move));
                    double val = alphaBeta(child, curDepth+1, maxDepth, alpha, beta, false);
                    bestVal = Math.max(bestVal, val);
                    alpha = Math.max(alpha, bestVal);
                    if(alpha>=beta) break;
                } catch(InvalidBotException e){}
            }
        } else {
            bestVal = Double.POSITIVE_INFINITY;
            for(MoveInfo mi: moves){
                if(mi.heuristic == Double.NEGATIVE_INFINITY) continue;
                try {
                    Board child = board.playMoveSimulationBoard(board.getCurrentPlayer(), moveVector(mi.move));
                    double val = alphaBeta(child, curDepth+1, maxDepth, alpha, beta, true);
                    bestVal = Math.min(bestVal, val);
                    beta = Math.min(beta, bestVal);
                    if(alpha>=beta) break;
                } catch(InvalidBotException e){}
            }
        }

        // store in TT
        TTEntry newEntry = new TTEntry();
        newEntry.value = bestVal;
        if(bestVal <= alpha){
            // upper bound
            newEntry.upperBound = bestVal;
            newEntry.lowerBound = Double.NEGATIVE_INFINITY;
        } else if(bestVal >= beta){
            // lower bound
            newEntry.lowerBound = bestVal;
            newEntry.upperBound = Double.POSITIVE_INFINITY;
        } else {
            // exact
            newEntry.upperBound = bestVal;
            newEntry.lowerBound = bestVal;
        }
        transposition.put(key, newEntry);

        return bestVal;
    }

    //===================== Move ordering plus fin =======================
    /**
     * Considère :
     *  - capture immediate
     *  - Q(s) max
     *  - seeds balance
     *  - envoi seeds
     */
    private List<MoveInfo> buildMoveInfos(Board board, int player){
        List<MoveInfo> list = new ArrayList<>();
        for(int move=0; move<Board.NB_HOLES; move++){
            try{
                int capture = board.playMoveSimulationScore(player, moveVector(move));
                if(capture<0){
                    list.add(new MoveInfo(move, Double.NEGATIVE_INFINITY));
                } else {
                    Board next = board.playMoveSimulationBoard(player, moveVector(move));
                    double cBonus = capture * CAPTURE_BONUS;
                    double qv = getMaxQVal(next);
                    double seedsEval = evaluateSeeds(next, player); // ex. combien de seeds tot

                    // On check combien on "envoie" à l'adversaire => ex. si trous adv reçoivent des seeds
                    // On peut approx si next.getOpponentHoles() sum big => penalty
                    int sumOpp = Arrays.stream(next.getOpponentHoles()).sum();
                    double sendPenalty = sumOpp * SEND_RISK_PENALTY * 0.01; // ajuster ?

                    double heur = cBonus + LAMBDA*qv + seedsEval*SEEDS_WEIGHT - sendPenalty;
                    list.add(new MoveInfo(move, heur));
                }
            } catch(InvalidBotException e){
                list.add(new MoveInfo(move, Double.NEGATIVE_INFINITY));
            }
        }
        return list;
    }

    private double evaluateSeeds(Board board, int player){
        // ex. sum of player holes
        int sumPlayer = Arrays.stream(board.getPlayerHoles()).sum();
        // normaliser
        return sumPlayer*0.1;
    }

    //===================== Evaluate final node =======================
    private boolean isTerminal(Board board){
        return board.getScore(0)>=25 || board.getScore(1)>=25 || board.getNbSeeds()<=6;
    }

    private double evaluate(Board board){
        // vantage = score diff
        double vantage = board.getScore(rootPlayer) - board.getScore(1-rootPlayer);

        // Q
        double qv = getMaxQVal(board);

        return vantage + LAMBDA*qv;
    }

    private double getMaxQVal(Board board){
        long code = encodeBoard(board);
        double[] arr = qTable.getOrDefault(code, new double[6]);
        return Arrays.stream(arr).max().orElse(0);
    }

    //===================== Encodage =======================
    private long encodeObservation(AweleObservation obs){
        // si on a la poss de partial capture => on l'a déjà dans la reward
        return encodeHoles(obs.getPlayerHoles(), obs.getOppenentHoles());
    }

    private long encodeBoard(Board board){
        return encodeHoles(board.getPlayerHoles(), board.getOpponentHoles());
    }

    private long encodeHoles(int[] ph, int[] oh){
        long code=0;
        for(int x: ph) {
            code = code*BASE_ENCODE + x;
        }
        for(int x: oh) {
            code = code*BASE_ENCODE + x;
        }
        return code;
    }

    private double[] moveVector(int move){
        double[] mv = new double[Board.NB_HOLES];
        mv[move]=1.0;
        return mv;
    }

    //===================== Structures internes =======================
    private static class MoveInfo {
        int move;
        double heuristic;
        MoveInfo(int m, double h){ move=m; heuristic=h;}
    }

    private static class TTKey {
        long stateCode;
        int depth;
        boolean isMax;
        int currentPlayer;

        TTKey(long c, int d, boolean im, int cp){
            stateCode=c; depth=d; isMax=im; currentPlayer=cp;
        }
        @Override
        public boolean equals(Object o){
            if(!(o instanceof TTKey)) return false;
            TTKey k=(TTKey)o;
            return (this.stateCode==k.stateCode && this.depth==k.depth && this.isMax==k.isMax && this.currentPlayer==k.currentPlayer);
        }
        @Override
        public int hashCode(){
            return Objects.hash(stateCode, depth, isMax, currentPlayer);
        }
    }
    private static class TTEntry {
        double value;
        double lowerBound=Double.NEGATIVE_INFINITY;
        double upperBound=Double.POSITIVE_INFINITY;
    }
}
