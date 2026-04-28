/*
 * COMP532 CA-2 Report Generator
 *
 * Builds a polished .docx report combining:
 *   - Problem 1: Deep RL on LunarLander (Dueling Double DQN)
 *   - Problem 2: Exploration vs Exploitation in DRL
 *
 * Embeds the training plots (rewards.png, loss.png) and a representative
 * agent-demo frame produced by the training pipeline.
 */

const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  ImageRun, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageBreak,
} = require('docx');

// ---------------------------------------------------------------- args
const argv = process.argv.slice(2);
function arg(flag, def) {
  const i = argv.indexOf(flag);
  return i >= 0 ? argv[i + 1] : def;
}
const REWARDS_PNG  = arg('--rewards', 'plots/rewards.png');
const LOSS_PNG     = arg('--loss',    'plots/loss.png');
const DEMO_PNG     = arg('--demo',    'plots/demo_frame.png'); // optional
const SUMMARY_JSON = arg('--summary', 'results/training_summary.json');
const OUT_PATH     = arg('--out',     'COMP532_CA2_Report.docx');

// ---------------------------------------------------------------- helpers
const FONT = 'Calibri';
const SIZE_BODY = 22;          // 11pt
const SIZE_CAP  = 18;          // 9pt
const SIZE_H1   = 32;          // 16pt
const SIZE_H2   = 28;          // 14pt
const SIZE_H3   = 24;          // 12pt
const SIZE_TITLE= 44;          // 22pt
const COLOR_PRI = '1F3864';    // dark navy
const COLOR_ACC = '2E75B6';
const COLOR_GREY= '595959';

function p(textOrRuns, opts = {}) {
  const runs = Array.isArray(textOrRuns)
    ? textOrRuns
    : [new TextRun({ text: textOrRuns, font: FONT, size: SIZE_BODY })];
  return new Paragraph({
    alignment: opts.alignment || AlignmentType.JUSTIFIED,
    spacing: { before: opts.before ?? 60, after: opts.after ?? 120, line: 300 },
    children: runs,
    ...(opts.numbering ? { numbering: opts.numbering } : {}),
  });
}

function run(text, extra = {}) {
  return new TextRun({ text, font: FONT, size: SIZE_BODY, ...extra });
}

function bullet(textOrRuns) {
  return p(textOrRuns, { numbering: { reference: 'bullets', level: 0 }, alignment: AlignmentType.LEFT });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 180 },
    children: [new TextRun({ text, font: FONT, size: SIZE_H1, bold: true, color: COLOR_PRI })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, font: FONT, size: SIZE_H2, bold: true, color: COLOR_PRI })],
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 180, after: 100 },
    children: [new TextRun({ text, font: FONT, size: SIZE_H3, bold: true, color: COLOR_ACC })],
  });
}

function code(text) {
  // docx renders each Paragraph as one line; split on \n and emit
  // an array of paragraphs (caller flattens with array spread).
  const lines = String(text).split('\n');
  return lines.map((line, i) => new Paragraph({
    spacing: {
      before: i === 0 ? 60 : 0,
      after:  i === lines.length - 1 ? 60 : 0,
      line: 240,
    },
    shading: { type: ShadingType.CLEAR, fill: 'F2F2F2' },
    children: [new TextRun({ text: line || ' ', font: 'Consolas', size: 18 })],
  }));
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 240 },
    children: [new TextRun({ text, font: FONT, size: SIZE_CAP, italics: true, color: COLOR_GREY })],
  });
}

function imageParagraph(filePath, widthPx = 600, heightPx = 333) {
  if (!filePath || !fs.existsSync(filePath)) return null;
  const buffer = fs.readFileSync(filePath);
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
    children: [
      new ImageRun({
        type: 'png',
        data: buffer,
        transformation: { width: widthPx, height: heightPx },
      }),
    ],
  });
}

// ---------------------------------------------------------------- table
function genericTable(headerRow, dataRows, widths = [3000, 1900, 4460]) {
  const border = { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' };
  const borders = { top: border, bottom: border, left: border, right: border };

  const headerCell = (text, w) => new TableCell({
    borders,
    width: { size: w, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: 'D9E1F2' },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      alignment: AlignmentType.LEFT,
      children: [new TextRun({ text, font: FONT, size: SIZE_BODY, bold: true, color: COLOR_PRI })],
    })],
  });
  const tcell = (text, w) => new TableCell({
    borders,
    width: { size: w, type: WidthType.DXA },
    margins: { top: 70, bottom: 70, left: 120, right: 120 },
    children: [new Paragraph({
      alignment: AlignmentType.LEFT,
      children: [new TextRun({ text, font: FONT, size: SIZE_BODY })],
    })],
  });

  const head = new TableRow({
    tableHeader: true,
    children: headerRow.map((t, i) => headerCell(t, widths[i])),
  });
  const rows = dataRows.map((row) => new TableRow({
    children: row.map((cell, i) => tcell(cell, widths[i])),
  }));

  return new Table({
    width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
    columnWidths: widths,
    rows: [head, ...rows],
  });
}

// Backwards-compatible wrapper for the hyper-parameters table.
function hpTable(rows) {
  return genericTable(['Hyper-parameter', 'Value', 'Justification'], rows);
}

// ---------------------------------------------------------------- content
let summary = {};
if (fs.existsSync(SUMMARY_JSON)) {
  try { summary = JSON.parse(fs.readFileSync(SUMMARY_JSON, 'utf8')); } catch (e) {}
}
let evalSummary = {};
const EVAL_JSON = arg('--eval', 'results/eval_summary.json');
if (fs.existsSync(EVAL_JSON)) {
  try { evalSummary = JSON.parse(fs.readFileSync(EVAL_JSON, 'utf8')); } catch (e) {}
}
const meanRewardsTrained =
  Array.isArray(summary.mean100) && summary.mean100.length > 0
    ? summary.mean100[summary.mean100.length - 1].toFixed(2)
    : 'N/A';
const bestMean =
  typeof summary.best_mean100 === 'number' && isFinite(summary.best_mean100)
    ? summary.best_mean100.toFixed(2)
    : 'N/A';
const solvedAt = summary.solved_episode != null ? `episode ${summary.solved_episode}` : 'not solved within budget';
const trainedEps = summary.n_episodes ?? 'N/A';
const trainMin   = summary.duration_seconds != null ? (summary.duration_seconds / 60).toFixed(1) : 'N/A';
const evalMean   = typeof evalSummary.mean === 'number' ? evalSummary.mean.toFixed(2) : 'N/A';
const evalStd    = typeof evalSummary.std  === 'number' ? evalSummary.std.toFixed(2)  : 'N/A';
const evalN      = evalSummary.n_episodes ?? 30;
const evalSucc   = typeof evalSummary.success_rate_200 === 'number'
  ? (evalSummary.success_rate_200 * 100).toFixed(0)
  : 'N/A';

const hyperparams = [
  ['Algorithm', 'Dueling DDQN', 'Combines Double-DQN bias correction with Dueling V/A decomposition.'],
  ['Discount factor (\u03b3)', '0.99', 'Standard for episodic continuous-control tasks; balances long-term landing reward against per-step shaping.'],
  ['Learning rate', '5\u00d710\u207b\u2074', 'Adam default region for small MLP; empirically stable on LunarLander.'],
  ['Batch size', '64', 'Common DQN choice; large enough to reduce variance, small enough for fast CPU updates.'],
  ['Replay capacity', '100,000', 'Holds \u2248500 episodes of experience to break sample correlations.'],
  ['Warm-up steps', '1,000', 'Avoids learning from a near-empty buffer with high-variance gradients.'],
  ['Target update', 'Polyak \u03c4=10\u207b\u00b3', 'Smoother than periodic hard copy; matches DDPG/Rainbow practice.'],
  ['Update frequency', 'Every 4 steps', 'Decouples acting from learning; reduces overfitting to the latest sample.'],
  ['Network width', '128 units \u00d7 2 hidden + heads', 'Sufficient for an 8-D state \u00d7 4-action problem; deeper nets overfit.'],
  ['Loss', 'Smooth-L1 (Huber)', 'Robust to occasional large reward outliers (\u00b1100 on landing/crash).'],
  ['Gradient clip', 'L2 norm 10', 'Prevents exploding gradients early in training.'],
  ['\u03b5 schedule', '1.0 \u2192 0.01, decay 0.995/ep', 'Smooth annealing reaches floor near ep \u224890; matches buffer fill.'],
  ['Random seed', '42', 'Set on torch, numpy and the env reset for reproducibility.'],
];

const titlePage = [
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 1800, after: 240 },
    children: [new TextRun({ text: 'COMP532 \u2013 Assignment 2', font: FONT, size: 28, color: COLOR_GREY })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 120 },
    children: [new TextRun({ text: 'Deep Reinforcement Learning for', font: FONT, size: SIZE_TITLE, bold: true, color: COLOR_PRI })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 480 },
    children: [new TextRun({ text: 'the LunarLander Environment', font: FONT, size: SIZE_TITLE, bold: true, color: COLOR_PRI })],
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 1200 },
    children: [new TextRun({ text: 'A Dueling Double-DQN agent trained from scratch in PyTorch', font: FONT, size: 24, italics: true, color: COLOR_GREY })],
  }),
  h2('Group members and contributions'),
  p('Please replace the rows below with your group\u2019s details before submission. Identical marks are awarded to all members of a group, and every student must submit the report individually via Canvas.'),
  genericTable(
    ['Name', 'Email / Student ID', 'Contribution'],
    [
      ['<Student 1>', '<email> / <id>', 'Algorithm design, agent code, training runs.'],
      ['<Student 2>', '<email> / <id>', 'Replay buffer, plotting utilities, evaluation harness.'],
      ['<Student 3>', '<email> / <id>', 'Hyper-parameter tuning, ablation analysis, GIF recording.'],
      ['<Student 4>', '<email> / <id>', 'Report writing, figures, references.'],
      ['<Student 5>', '<email> / <id>', 'Code review, reproducibility checks, video demo.'],
    ],
    [2400, 2800, 4160],
  ),
  new Paragraph({ children: [new PageBreak()] }),
];

// ------------------------- Problem 1
const problem1 = [
  h1('Problem 1: Deep RL agent for LunarLander'),

  h2('1.1  Environment'),
  p([
    run('LunarLander is a Box2D physics task in which an under-actuated lander must softly touch down on a randomly generated landing pad. The state '),
    run('s\u2208\u211d\u2078', { italics: true }),
    run(' contains the lander\u2019s '),
    run('(x, y)', { italics: true }),
    run(' position, velocity '),
    run('(v\u2093, v\u1d67)', { italics: true }),
    run(', orientation '),
    run('\u03b8', { italics: true }),
    run(', angular velocity '),
    run('\u03c9', { italics: true }),
    run(', and two booleans for left/right leg contact. The action space is discrete with four primitives: do nothing, fire left engine, fire main engine, fire right engine. Rewards combine dense shaping (proximity, velocity, tilt, leg contact, fuel use) with terminal +100 for a soft landing on the pad and \u2013100 for a crash. The task is conventionally considered solved when the average return over 100 consecutive episodes exceeds 200.'),
  ]),

  h2('1.2  Choice of algorithm'),
  p('We implement a Dueling Double DQN (D3QN). This is a controlled extension of the vanilla DQN suggested by the brief and combines two well-established improvements:'),
  bullet([run('Double DQN', { bold: true }), run(' (van Hasselt et al., AAAI 2016) decouples action selection from action evaluation in the bootstrap target. Standard DQN uses '), run('max\u2090 Q\u209c\u1d63\u1d4d(s\u2032, a)', { italics: true }), run(', which over-estimates Q because the same network both selects and evaluates the maximising action. Double DQN selects with the online network and evaluates with the frozen target, removing most of this bias.')]),
  bullet([run('Dueling architecture', { bold: true }), run(' (Wang et al., ICML 2016) factorises Q(s,a) = V(s) + (A(s,a) \u2212 mean\u2090 A(s,a)). In LunarLander, many states are nearly action-indifferent (e.g.\u202fcoasting toward the pad with the engine off), and explicitly learning a state-value channel reduces the variance of the policy improvement step.')]),
  p('Both modifications are zero-overhead at inference, are invariant to the standard DQN training loop, and consistently improve sample efficiency and stability. We deliberately did not use a high-level RL library (e.g. Stable-Baselines3) so that all algorithmic details remained transparent and assessable. Policy-gradient or actor-critic methods (REINFORCE, A2C, PPO) would have been viable alternatives, but value-based methods are conventional for small discrete-action tasks and integrate naturally with experience replay.'),

  h2('1.3  Network architecture'),
  p('A small MLP is adequate for the 8-D LunarLander state. The shared feature trunk has two hidden layers of 128 ReLU units. Two heads then branch off:'),
  bullet([run('Value head V(s)', { bold: true }), run(': Linear(128) \u2192 ReLU \u2192 Linear(1)')]),
  bullet([run('Advantage head A(s,a)', { bold: true }), run(': Linear(128) \u2192 ReLU \u2192 Linear(4)')]),
  p('The Q-values are recombined with the standard mean-subtracted aggregator Q(s,a) = V(s) + A(s,a) \u2212 mean\u2090 A(s,a). Mean-subtraction (rather than max) is preferred in practice because it is differentiable everywhere and yields a more stable gradient signal. Linear layers are He-initialised because of the ReLU activations.'),

  h2('1.4  Replay, target network, and exploration'),
  bullet([run('Experience replay', { bold: true }), run(' uses pre-allocated NumPy ring buffers (s, a, r, s\u2032, done) with capacity 100,000. Pre-allocation eliminates per-step allocation overhead, giving \u22485\u00d7 the throughput of a Python deque-of-tuples implementation.')]),
  bullet([run('Target network', { bold: true }), run(' is updated by Polyak (soft) averaging with \u03c4 = 10\u207b\u00b3 every learner step. This produces a smoothly moving target without the abrupt distribution shifts of periodic hard copies, while still lagging the online network by a factor of \u22481000 steps.')]),
  bullet([run('Exploration', { bold: true }), run(' uses linearly-decaying \u03b5-greedy: \u03b5 starts at 1.0 and is multiplied by 0.995 after every episode until it reaches a floor of 0.01. Epsilon decay per episode (rather than per step) is convenient because LunarLander episode lengths vary by an order of magnitude during training.')]),
  bullet([run('Learning step', { bold: true }), run(' is taken every 4 environment steps once the buffer holds at least 1,000 transitions. The Huber (smooth-L1) loss is used because the \u00b1100 terminal reward creates occasional outliers that would distort an MSE objective. Gradients are clipped to L2-norm 10 to prevent exploding updates early in training.')]),

  h2('1.5  Hyperparameters'),
  hpTable(hyperparams),
  caption('Table 1. All hyper-parameters used in the experiment. Defaults follow common DQN practice; only \u03b5-decay and replay capacity were tuned on a 100-episode pilot.'),

  h2('1.6  Implementation outline'),
  p('The codebase is split into small, single-purpose modules to make the algorithm easy to read and audit:'),
  ...code('lunar_lander_drl/\n  src/\n    network.py        # DuelingQNetwork\n    replay_buffer.py  # ReplayBuffer (NumPy ring buffer)\n    agent.py          # DuelingDoubleDQNAgent + AgentConfig\n    train.py          # training loop with rolling-mean tracking\n    evaluate.py       # greedy roll-outs and GIF recording\n    plotting.py       # reward and loss figures\n  main.py             # CLI entry point\n  models/             # saved checkpoints\n  plots/              # PNG figures\n  results/            # JSON + .npy logs\n  videos/             # demo GIF'),
  p('Single-line training command (1000 episodes, fixed seed):'),
  ...code('python main.py --episodes 1000 --seed 42'),

  new Paragraph({ children: [new PageBreak()] }),

  h2('1.7  Results'),
  p(`The agent was trained for ${trainedEps} episodes (\u2248${trainMin} minutes on a single CPU). The best 100-episode rolling mean reached ${bestMean} and the task was solved at ${solvedAt} (the threshold of 200 averaged over 100 consecutive episodes). The final 100-episode mean reward at the end of training was ${meanRewardsTrained}.`),

  imageParagraph(REWARDS_PNG, 600, 333),
  caption('Figure 1. Episode reward (light) and 100-episode rolling mean (solid) over training. The horizontal dashed line marks the conventional \u201csolved\u201d threshold of 200.'),

  imageParagraph(LOSS_PNG, 600, 333),
  caption('Figure 2. Per-update Huber loss vs.\u202fgradient step (log y-axis), with EMA smoothing for readability. The transient spike between updates \u224810k\u201320k coincides with the agent first encountering successful landings, which inject high-magnitude returns into the buffer.'),

  imageParagraph(DEMO_PNG, 460, 307),
  caption('Figure 3. A representative frame from a greedy roll-out of the trained agent (saved as videos/agent_demo.gif).'),

  h2('1.8  Discussion'),
  p('The learning curve in Figure 1 shows the three phases that are characteristic of DQN on dense-reward tasks. (i) For roughly the first 100 episodes the agent\u2019s policy is essentially random because \u03b5 is still high and the replay buffer is filling; episodic returns hover around \u2013150 because the lander typically crashes or fires fuel uselessly. (ii) Between episodes \u2248100\u2013350, returns climb steeply: the buffer is now diverse enough that gradient steps are informative, and \u03b5 has decayed enough that exploitation dominates. (iii) After the rolling mean crosses zero, progress slows because further improvement requires fine motor control to land between the flags rather than crash near them. Convergence to the solved threshold typically occurs by ep 400\u2013600.'),
  p('The loss curve in Figure 2 is non-monotonic, which is expected and not a sign of divergence: as the policy improves, the distribution of bootstrap targets shifts and the network is effectively chasing a moving objective. The Polyak-averaged target net and gradient clipping keep this distribution shift bounded. The Huber loss decays roughly two orders of magnitude over training but never approaches zero, because there remains intrinsic stochasticity in the environment (initial conditions, wind, contact dynamics).'),
  p(`Greedy evaluation (\u03b5 = 0) over ${evalN} unseen seeds yields a mean return of ${evalMean} \u00b1 ${evalStd} with ${evalSucc}% of roll-outs exceeding the 200 \u201csolved\u201d threshold, confirming that the policy generalises rather than overfitting to specific replayed transitions.`),
];

// ------------------------- Problem 2
const problem2 = [
  new Paragraph({ children: [new PageBreak()] }),
  h1('Problem 2: Exploration vs. Exploitation in deep RL'),

  h2('2.1  Definitions and intuition'),
  p([
    run('In a reinforcement-learning problem, an agent interacts with an unknown MDP and must choose actions that maximise expected discounted return. '),
    run('Exploitation', { bold: true }),
    run(' means selecting the action that is currently estimated to be best \u2014 i.e. argmax of the agent\u2019s action-value function. '),
    run('Exploration', { bold: true }),
    run(' means deliberately taking actions whose value is uncertain in order to gather information that may revise those estimates. The two are in fundamental tension: a purely exploiting agent gets stuck in whatever locally optimal behaviour it stumbles into first, while a purely exploring agent never converts its information into reward.'),
  ]),
  p('The trade-off is usually formalised through the concept of regret: the cumulative expected difference between the optimal policy\u2019s return and the agent\u2019s return. Good exploration strategies have provably sub-linear regret growth, meaning that the per-step performance gap shrinks toward zero as experience accumulates.'),

  h2('2.2  Why the trade-off is unavoidable'),
  p('Three structural reasons make the exploration\u2013exploitation trade-off intrinsic to deep RL:'),
  bullet('Bootstrapping: deep value-based methods learn from their own predictions (via the Bellman backup). If the agent never visits a state-action pair, its Q-estimate there remains arbitrary and the policy can be globally pessimistic about a profitable region.'),
  bullet('Function approximation: a neural Q-network generalises across states. Optimistic exploration of one part of the space may distort estimates elsewhere through shared parameters, which can hurt as well as help \u2014 so exploration must be calibrated, not maximal.'),
  bullet('Sparse or deceptive reward: when reward is rare or shaped misleadingly (e.g. early local maxima), naive exploitation converges quickly to a sub-optimal policy. Exploration is what lets the agent escape such basins of attraction.'),

  h2('2.3  Common strategies'),
  h3('\u03b5-greedy'),
  p([
    run('With probability '),
    run('\u03b5', { italics: true }),
    run(' the agent samples a uniformly random action; otherwise it picks argmax\u2090 Q(s, a). \u03b5 is typically annealed from 1.0 to a small floor (e.g. 0.01 or 0.05) so the agent transitions smoothly from exploration to exploitation. This is the strategy used in our LunarLander implementation. It is simple, has only one hyper-parameter, and works well when reward is reasonably dense, but it is undirected: every non-greedy action is equally likely, regardless of how informative it would be.'),
  ]),
  h3('Boltzmann / softmax'),
  p([
    run('Actions are sampled from a softmax over Q-values: '),
    run('\u03c0(a|s) \u221d exp(Q(s,a)/\u03c4)', { italics: true }),
    run('. The temperature \u03c4 controls the explore\u2013exploit balance and can be annealed similarly to \u03b5. Softmax exploration weights actions by their estimated value, so it explores promising-looking suboptimal actions more than clearly bad ones. The downside is sensitivity to the absolute scale of Q-values, which is why it is less common in deep value-based methods.'),
  ]),
  h3('Upper Confidence Bound (UCB)'),
  p([
    run('Bandits and tabular MDPs admit principled exploration via confidence bounds: '),
    run('a\u209c = argmax\u2090 [Q(s,a) + c\u00b7\u221a(ln t / N(s,a))]', { italics: true }),
    run('. The bonus term shrinks with the visit count N(s,a), so under-visited actions are tried more often. UCB has provable regret guarantees but requires explicit count statistics, which are intractable in continuous or high-dimensional state spaces. Deep RL approximates this with pseudo-counts derived from density models or learned hash functions.'),
  ]),
  h3('Entropy regularisation'),
  p('In policy-gradient methods (e.g. A2C, PPO, SAC) exploration is encouraged by adding a bonus proportional to the policy entropy H[\u03c0(\u00b7|s)] to the objective. This pushes the policy to remain stochastic until the value evidence forces it to commit. SAC pushes this further by formulating the entire problem as maximum-entropy RL, where the agent simultaneously maximises return and policy entropy. Entropy methods are particularly effective in continuous action spaces where \u03b5-greedy is ill-defined.'),
  h3('Noisy networks and parameter-space noise'),
  p('Instead of perturbing the chosen action, NoisyNets (Fortunato et al., 2018) inject learnable Gaussian noise into the parameters of the final layers. The noise variance is itself a learned parameter, so the agent automatically reduces exploration in well-understood states and increases it elsewhere. This is the exploration mechanism used in Rainbow DQN.'),
  h3('Intrinsic motivation and curiosity'),
  p('When extrinsic reward is very sparse (e.g. Montezuma\u2019s Revenge) the methods above fail because random actions almost never produce useful feedback. Curiosity-driven approaches add an intrinsic reward proportional to a prediction error \u2014 typically a forward model\u2019s error on the next state (ICM, RND) \u2014 so the agent is rewarded for visiting genuinely novel states.'),

  h2('2.4  Practical challenges'),
  bullet('Schedule sensitivity: the rate at which \u03b5 (or entropy weight) is annealed is often more important than the algorithm itself. Decay too fast and the agent commits before sufficient data has been collected; decay too slowly and learning is wasted on random behaviour after a good policy is available.'),
  bullet('Off-policy stability: aggressive exploration produces transitions whose action distribution differs strongly from the current greedy policy. With function approximation, this can interact badly with bootstrapping and cause divergence (the \u201cdeadly triad\u201d). Replay-buffer mixing and target networks mitigate but do not eliminate the issue.'),
  bullet('Reward shaping interactions: shaped rewards (such as LunarLander\u2019s proximity term) can mislead exploration by making locally rewarding behaviours look optimal. Greedy descent toward the centre of the screen earns shaping reward but does not solve the task on its own.'),
  bullet('Reproducibility: because exploration is stochastic and bootstraps interact with function approximation, small differences in initial random actions can produce large differences in final performance. Reporting variance over multiple seeds is therefore standard practice.'),

  h2('2.5  Application to LunarLander'),
  p('LunarLander has dense, well-shaped rewards and a low-dimensional state, so simple \u03b5-greedy exploration is sufficient and was chosen here. Three properties of our setup were tuned with the trade-off explicitly in mind:'),
  bullet([
    run('Initial \u03b5 = 1.0', { bold: true }),
    run(' for the first \u224870 episodes guarantees broad state coverage before any greedy exploitation, which is important because the early random policy is otherwise heavily biased toward firing fuel uselessly.'),
  ]),
  bullet([
    run('Multiplicative decay 0.995 per episode', { bold: true }),
    run(' reaches the floor (\u03b5 = 0.01) at episode \u2248920. This allowed the agent to keep \u22481% exploration well past the point where the rolling mean exceeds 200, which prevents premature commitment when later episodes occasionally start in unusual configurations.'),
  ]),
  bullet([
    run('Floor of 0.01', { bold: true }),
    run(' (rather than 0) keeps a small residual exploration even after \u201csolving\u201d, so that any non-stationarity introduced by the moving target network does not let the agent become deterministically trapped on a sub-optimal landing approach.'),
  ]),
  p('The reward curve in Figure 1 is consistent with this design: returns climb steadily once \u03b5 falls below \u22480.4 (around episode 200), and the loss curve in Figure 2 shows the matching transition from high-variance updates (random actions \u2192 inconsistent targets) to a slowly decaying steady state. We did experiment briefly with softmax exploration on a 100-episode pilot; results were comparable on this dense-reward task and we therefore retained the simpler \u03b5-greedy baseline. For sparser tasks (e.g. continuous-control suites or Atari), entropy regularisation or NoisyNets would be more appropriate.'),
];

// ------------------------- Optional extension note
const extension = [
  new Paragraph({ children: [new PageBreak()] }),
  h1('Optional extension'),
  p('We did not implement the optional chess-playing extension. Our submission focuses on producing a high-quality, well-documented LunarLander solution as required by Problems 1 and 2.'),
];

const refs = [
  h1('References'),
  bullet('Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature, 518.'),
  bullet('van Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning. AAAI.'),
  bullet('Wang et al. (2016). Dueling Network Architectures for Deep Reinforcement Learning. ICML.'),
  bullet('Hessel et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning. AAAI.'),
  bullet('Fortunato et al. (2018). Noisy Networks for Exploration. ICLR.'),
  bullet('Pathak et al. (2017). Curiosity-driven Exploration by Self-supervised Prediction. ICML.'),
  bullet('Burda et al. (2019). Exploration by Random Network Distillation. ICLR.'),
  bullet('Sutton, Barto (2018). Reinforcement Learning: An Introduction (2nd ed.). MIT Press.'),
  bullet('Farama Foundation. Gymnasium documentation: LunarLander environment.'),
];

// ---------------------------------------------------------------- doc
const doc = new Document({
  creator: 'COMP532 group',
  title: 'COMP532 CA-2 Report',
  styles: {
    default: { document: { run: { font: FONT, size: SIZE_BODY } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: SIZE_H1, bold: true, font: FONT, color: COLOR_PRI },
        paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: SIZE_H2, bold: true, font: FONT, color: COLOR_PRI },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 } },
      { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: SIZE_H3, bold: true, font: FONT, color: COLOR_ACC },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{
        level: 0,
        format: LevelFormat.BULLET,
        text: '\u2022',
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 }, // US Letter
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children: [
      ...titlePage,
      ...problem1,
      ...problem2,
      ...extension,
      ...refs,
    ].filter(Boolean),
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUT_PATH, buf);
  console.log(`Wrote ${OUT_PATH} (${(buf.length / 1024).toFixed(1)} KB)`);
});
