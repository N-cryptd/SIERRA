# SIERRA
Survival Intelligence Environment for Resource Research and Adaptation

## Requirements Analysis

### Functional Requirements:

- Grid-based navigation environment with multi-resource management (food, water, materials)
- Dynamic environmental conditions (day/night cycles, weather patterns, seasons)
- Threat and hazard system with avoidance mechanics
- **Crafting mechanics for creating tools, shelter, or other essential items**
- Reinforcement learning agent implementation with observation and action spaces
- Training framework with configurable hyperparameters
- Visualization system for game states and agent performance
- Metrics collection and analysis for learning evaluation
- Save/load functionality for trained models and training sessions

### Non-Functional Requirements:

- Performance: Simulation speed of at least 1000 steps/second for efficient training
- Scalability: Support for parallel training instances
- Reliability: Stable execution for extended training periods (10M+ steps)
- Observability: Comprehensive metrics dashboard and behavior visualization
- Maintainability: Modular code structure with clear separation of concerns

## Architecture Overview

```mermaid
graph LR
    subgraph "Core Gameplay Loop"
        EC[Environment Core<br/>Game Logic, State Management]
        AF[Agent Framework<br/>DRL Policy, Neural Networks]
        TM[Training Manager<br/>Training Loop, Hyperparameters]
    end

    subgraph "Supporting Systems"
        VE[Visualization Engine<br/>Game State Rendering]
        DCS[Data Collection System<br/>Metrics, Performance Analysis]
        MR[Model Repository<br/>Model Storage, Versioning]
    end

    EC -- State Observations --> AF
    AF -- Actions --> EC
    EC -- Rewards --> TM
    TM -- Update Policy --> AF
    EC -- Game State --> VE
    TM -- Performance Data --> DCS
    TM -- Trained Models --> MR

    %% Styling (Optional - matches original colors somewhat)
    style EC fill:#cceeff,stroke:#333,stroke-width:2px
    style AF fill:#ccffcc,stroke:#333,stroke-width:2px
    style TM fill:#ffebcc,stroke:#333,stroke-width:2px
    style VE fill:#e6ccff,stroke:#333,stroke-width:2px
    style DCS fill:#ccccff,stroke:#333,stroke-width:2px
    style MR fill:#ffcccc,stroke:#333,stroke-width:2px
```

### Component Descriptions

- **Environment Core**: Game logic, state transitions, and reward calculation
- **Agent Framework**: Deep RL implementation with neural network policies
- **Training Manager**: Coordinates training sessions and hyperparameter optimization
- **Visualization Engine**: Renders game state and agent behavior for analysis
- **Data Collection System**: Gathers performance metrics and learning statistics
- **Model Repository**: Stores and versions trained models

### Data Flow Patterns

```mermaid
graph TD
    A[Agent<br/>Neural Network Policy]
    E[Environment<br/>Game World]
    TS[Training System<br/>Experience Collection]
    MB[Memory Buffer<br/>Experience Replay]
    MS[Metrics System]

    E -- State St --> A
    A -- Action At --> E
    E -- "(St,At)" --> TS
    E -- Rt --> TS
    TS -- "(St,At,Rt,St+1)" --> MB
    MB -- Experience Batch --> A
    TS -- Policy Updates --> A
    MB -- Performance Data --> MS
    TS -- Learning Metrics --> MS

    %% Styling to somewhat match original colors
    style A fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
    style E fill:#e1f5fe,stroke:#0288d1,stroke-width:2px
    style TS fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    style MB fill:#e8eaf6,stroke:#3949ab,stroke-width:2px
    style MS fill:#fbe9e7,stroke:#d84315,stroke-width:2px

    %% Note: Mermaid doesn't easily support the 'Training Loop Cycle' arrow visually overlaid like in the diagram.
    %% This representation focuses on the direct data flow between components.
```

- Environment → Agent: State observations (resource levels, position, threats)
- Agent → Environment: Actions (movement, resource collection, crafting)
- Environment → Data Collector: Rewards, state transitions, episode statistics
- Training Manager → Agent: Hyperparameter updates and learning rate schedules

## Data Model

![Captura de pantalla 2025-04-27 060642](https://github.com/user-attachments/assets/f6be9bad-60f0-4cc5-91a8-9b386e4f78b8)


- **GameState**: Grid representation, agent position, resource locations, threat positions
- **AgentObservation**: Partial or full state information provided to agent
- **Action**: Discrete or continuous action space (movement, interaction, crafting)
- **Reward**: Multi-component reward signal based on survival metrics
- **TrainingMetadata**: Hyperparameters, episode counts, total steps

```mermaid
graph TD
    %% Title (as comment)
    %% Reward Function Components - Deep RL Resource Survival Game

    TR(("Total Reward<br/>R = α₁R₁ + α₂R₂ + α₃R₃ +<br/>α₄R₄ + α₅R₅"))

    subgraph "Reward Components"
        direction TB %% Changed direction to allow LP below others potentially
        SR(("Survival Reward<br/>R₁ = +0.1 per step<br/>alive"))
        RC(("Resource<br/>Collection<br/>R₂ = +0.5 per resource"))
        HM(("Health<br/>Management<br/>R₃ = -1.0 for damage"))
        EI(("Exploration<br/>Incentive<br/>R₄ = +0.01 per new cell"))
        LP(("Long-term<br/>Planning<br/>R₅ = +5.0 for shelter built"))
    end

    %% Connections with weights
    TR -- "α₁=0.2" --> SR
    TR -- "α₂=0.3" --> RC
    TR -- "α₃=0.2" --> HM
    TR -- "α₄=0.1" --> EI
    TR -- "α₅=0.2" --> LP

    %% Annotations (represented as separate nodes, unlinked as they are contextual)
    subgraph "Context & Principles"
        direction TB
        TP[/"Termination Penalty:<br/>-10.0 for death"/]
        DP["• Component weights (α) can be tuned...<br/>• Sparse rewards (R₅) provide long-term direction..."]
    end

    %% Styling
    style TR fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    style SR fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    style RC fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px
    style HM fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    style EI fill:#fff8e1,stroke:#ff8f00,stroke-width:2px
    style LP fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    style TP fill:#ffcdd2,stroke:#d32f2f,stroke-width:1px
    style DP fill:#f1f3f4,stroke:#ccc,stroke-width:1px

    %% Note: Mermaid will handle layout; exact positioning relative to original may vary.
    %% Annotations are grouped but not directly linked into the reward calculation flow.
```
## Technology Stack

- **Programming Language**: Python 3.9+
- **RL Frameworks**: PyTorch, Stable Baselines3
- **Environment**: Custom implementation with NumPy/PyGame
- **Data Processing**: NumPy, Pandas
- **Visualization**: Matplotlib, TensorBoard
- **Distributed Training**: Ray
- **Containerization**: Docker

## Implementation Roadmap

The development will follow an iterative process, visualized below. The following sequential milestones represent the primary focus within these iterative development cycles.

```mermaid
graph LR
    %% Title: Deep RL Development Cycle - Iterative Process for Resource Survival Game

    %% Central Hub
    IM(("<b>Iteration Management</b><br/>Version control, experiment<br/>tracking, documentation"))

    %% Main Cycle Stages with Embedded Tasks & Timing
    ED("<b>1. Environment Design</b><br/><small><i>(Week 1-2)</i></small><br/>Resource dynamics, state space, rewards<hr/><u>Tasks:</u><br/>• Define resource types<br/>• Implement dynamics<br/>• Design reward function")
    AA("<b>2. Agent Architecture</b><br/><small><i>(Week 3-5)</i></small><br/>Neural network structure, algorithm selection<hr/><u>Tasks:</u><br/>• Design network layers<br/>• Implement algorithms<br/>• Optimize observation")
    TP("<b>3. Training Process</b><br/><small><i>(Week 6-10)</i></small><br/>Hyperparameter tuning, learning schedules<hr/><u>Tasks:</u><br/>• Hyperparameter search<br/>• Reward shaping<br/>• Curriculum learning")
    EA("<b>4. Evaluation & Analysis</b><br/><small><i>(Week 11-14)</i></small><br/>Performance metrics, behavior assessment<hr/><u>Tasks:</u><br/>• Survival rate metrics<br/>• Strategy analysis<br/>• Scenario testing")

    %% Connections forming the cycle and linking to the hub
    IM --> ED
    ED --> AA
    AA --> TP
    TP --> EA
    EA --> ED 
    EA --> IM 

    %% Styling Nodes (approximating original colors and shapes)
    style IM fill:#f5f5f5,stroke:#616161,stroke-width:2px,text-align:center
    style ED fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,text-align:left %% Blue
    style AA fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,text-align:left %% Green
    style TP fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,text-align:left %% Orange
    style EA fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,text-align:left %% Purple

    %% Styling Links (color matches the source node)
    %% Indices: 0:IM->ED, 1:ED->AA, 2:AA->TP, 3:TP->EA, 4:EA->ED, 5:EA->IM
    linkStyle 0 stroke:#616161,stroke-width:2px;
    linkStyle 1 stroke:#1565c0,stroke-width:2px;
    linkStyle 2 stroke:#2e7d32,stroke-width:2px;
    linkStyle 3 stroke:#ef6c00,stroke-width:2px;
    linkStyle 4 stroke:#6a1b9a,stroke-width:2px;
    linkStyle 5 stroke:#6a1b9a,stroke-width:2px;

    %% Note: Mermaid handles layout automatically, which might not be a perfect circle.
    %% The dashed background circles and the 'i' icon are not directly representable.
```

1. Basic environment with single resource type (2 weeks)
2. Multi-resource management system (2 weeks)
3. Dynamic environmental conditions (2 weeks)
4. Basic RL agent integration (1 week)
5. Advanced agent architectures (PPO, SAC, etc.) (3 weeks)
6. Visualization and analysis tools (2 weeks)
7. Performance optimization (2 weeks)

## Integration Testing Framework

```mermaid
graph TD
    %% Title: Testing Framework Architecture - Deep RL Resource Survival Game

    %% Central Orchestration
    TOF[Test Orchestration Framework<br/>Automated Test Scheduling, Reporting, and Analysis]

    subgraph "Inputs & Outputs"
        direction TB
        TT["<b>Testing Tools</b><br/>• PyTest<br/>• Hypothesis<br/>• TensorBoard<br/>• Ray Test Tools"]
        TM["<b>Test Metrics</b><br/>• Coverage<br/>• Reward Targets<br/>• Performance<br/>• Training Stability"]
    end

    subgraph "Standard Test Types"
        direction TB
        UT["<b>Unit Tests</b><br/>Environment Components, Neural<br/>Network Layers, Utilities"]
        IT["<b>Integration Tests</b><br/>API Contracts, Component<br/>Interactions, Data Flows"]
        ST["<b>System Tests</b><br/>End-to-End Scenarios,<br/>Performance Benchmarks"]
    end

    subgraph "RL-Specific Tests"
        direction TB
        AT["<b>Algorithm Tests</b><br/>Loss Functions, Gradient Flow,<br/>Update Mechanisms"]
        PV["<b>Policy Validation</b><br/>Behavior Analysis, Entropy<br/>Checks, Action Distributions"]
        ScT["<b>Scenario Testing</b><br/>Survival Tests, Resource<br/>Management, Threat Reactions"]
    end

    subgraph "Execution & Integration"
        direction TB
        subgraph TE["Test Environments"]
            direction TB
            SGW[Simple Grid World]
            StdE[Standard Environment]
            CS[Challenge Scenarios]
        end
        CICD["<b>CI/CD Pipeline Integration</b><br/>GitHub Actions, Jenkins, Test Reports"]
    end

    %% Connections
    TT --> TOF
    TOF --> TM
    TOF --> UT
    TOF --> IT
    TOF --> ST
    TOF --> AT
    TOF --> PV
    TOF --> ScT
    TOF --> TE

    %% Dashed links to Test Environments
    UT -.-> TE
    IT -.-> TE
    ST -.-> TE
    AT -.-> TE
    PV -.-> TE
    ScT -.-> TE

    TE --> CICD

    %% Styling
    style TOF fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,color:#000
    style TT fill:#f1f8e9,stroke:#558b2f,stroke-width:1px,color:#000
    style TM fill:#ede7f6,stroke:#5e35b1,stroke-width:1px,color:#000
    style UT fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px,color:#000
    style IT fill:#fff3e0,stroke:#ef6c00,stroke-width:2px,color:#000
    style ST fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,color:#000
    style AT fill:#ffebee,stroke:#c62828,stroke-width:2px,color:#000
    style PV fill:#e0f7fa,stroke:#00acc1,stroke-width:2px,color:#000
    style ScT fill:#e8eaf6,stroke:#3949ab,stroke-width:2px,color:#000
    style TE fill:#fafafa,stroke:#90a4ae,stroke-width:2px,color:#000
    style SGW fill:#e1f5fe,stroke:#0288d1,stroke-width:1px,color:#000
    style StdE fill:#e1f5fe,stroke:#0288d1,stroke-width:1px,color:#000
    style CS fill:#e1f5fe,stroke:#0288d1,stroke-width:1px,color:#000
    style CICD fill:#f5f5f5,stroke:#616161,stroke-width:2px,color:#000

    %% Style dashed links (indices might need adjustment based on rendering order)
    linkStyle 10 stroke-dasharray: 5 5, stroke:#2e7d32,stroke-width:1px;
    linkStyle 11 stroke-dasharray: 5 5, stroke:#ef6c00,stroke-width:1px;
    linkStyle 12 stroke-dasharray: 5 5, stroke:#6a1b9a,stroke-width:1px;
    linkStyle 13 stroke-dasharray: 5 5, stroke:#c62828,stroke-width:1px;
    linkStyle 14 stroke-dasharray: 5 5, stroke:#00acc1,stroke-width:1px;
    linkStyle 15 stroke-dasharray: 5 5, stroke:#3949ab,stroke-width:1px;
```

### Testing Components

- Environment consistency tests
- Agent interface compatibility tests
- Training stability tests
- Performance benchmarks for various algorithms
- Reward signal validation


## RL Algorithm Comparison

Comparative experiments were conducted to evaluate the performance of PPO and DQN algorithms on the SIERRA environment. The key evaluation metrics are summarized below:

*   **PPO Model:**
    *   Mean Episode Reward: -21.28 +/- 2.91
    *   Mean Episode Length: 946.83
*   **DQN Model:**
    *   Mean Episode Reward: -42.81 +/- 11.10
    *   Mean Episode Length: 949.49

**Interpretation:**

The PPO model demonstrated significantly better performance compared to the DQN model, achieving a higher mean episode reward and a lower standard deviation, indicating more stable and successful learning. The mean episode lengths were comparable for both algorithms.

It's worth noting that the SAC algorithm was also considered but was found to be incompatible with the current discrete action space of the SIERRA environment.

## User Interface Framework

### Component Specifications

- Training configuration interface
- Real-time visualization of agent behavior
- Performance metrics dashboard
- Learning curve visualization
- Model comparison tools

## System Monitoring and Observability
![Captura de pantalla 2025-04-27 063524](https://github.com/user-attachments/assets/656c06e0-8a42-41b7-b49d-bb01e83ec2ad)


- Episode return tracking
- Resource management efficiency metrics
- Survival duration statistics
- Environment exploration coverage
- Policy entropy and value loss monitoring

## Infrastructure Requirements

### Infrastructure Sizing

- Development: 8+ CPU cores, 16GB+ RAM, NVIDIA GPU (4GB+ VRAM)
- Training: Cloud GPU instance or local workstation with RTX series GPU
- Storage: 50GB for code, models, and training data

### Cost Estimation (Monthly)

- Local development: Hardware depreciation + electricity (~$30/month)
- Cloud GPU training (optional): $100-300/month depending on instance type and usage

## Documentation Strategy

- Code documentation with docstrings
- README and setup instructions
- Training configuration guides
- Experiment results analysis
- Model performance comparisons
