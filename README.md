# Swarms Deploy 🚀

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


[![PyPI version](https://badge.fury.io/py/swarms-deploy.svg)](https://badge.fury.io/py/swarms-deploy)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Production-grade API deployment framework for Swarms AI workflows. Easily deploy, scale, and manage your swarm-based applications with enterprise features.

## Features ✨

- 🔥 Fast API-based deployment framework
- 🤖 Support for synchronous and asynchronous swarm execution
- 🔄 Built-in load balancing and scaling
- 📊 Real-time monitoring and logging
- 🛡️ Enterprise-grade error handling
- 🎯 Priority-based task execution
- 📦 Simple deployment and configuration
- 🔌 Extensible plugin architecture

## Installation 📦

```bash
pip install -U swarms-deploy
```

## Quick Start 🚀

```python
from swarms_deploy import SwarmDeploy
from swarms.workflows import SequentialWorkflow

# Initialize your agents
data_extractor_agent = YourDataExtractorAgent()
summarizer_agent = YourSummarizerAgent()
financial_analyst_agent = YourFinancialAnalystAgent()

# Create a workflow
workflow = SequentialWorkflow(
    name="document-analysis-swarm",
    description="Document analysis pipeline",
    agents=[
        data_extractor_agent,
        summarizer_agent,
        financial_analyst_agent
    ],
    output_type="all"
)

# Deploy the swarm
swarm = SwarmDeploy(workflow)
swarm.start(host="0.0.0.0", port=8000)
```

## Advanced Usage 🔧

### Configuration Options

```python
swarm = SwarmDeploy(
    workflow,
    max_workers=4,
    cache_backend="redis",
    ssl_config={
        "keyfile": "path/to/key.pem",
        "certfile": "path/to/cert.pem"
    }
)
```

### Clustering and Scaling

```python
# Create a distributed cluster
instances = SwarmDeploy.create_cluster(
    workflow,
    num_instances=3,
    start_port=8000,
    hosts=["host1", "host2", "host3"]
)
```

## API Reference 📚

### SwarmInput Model

```python
class SwarmInput(BaseModel):
    task: str          # Task description
    img: Optional[str] # Optional image input
    priority: int      # Task priority (0-10)
```

### API Endpoints

- **POST** `/v1/swarms/completions/{callable_name}`
  - Execute a task with the specified swarm
  - Returns: SwarmOutput or SwarmBatchOutput

### Example Request

```bash
curl -X POST "http://localhost:8000/v1/swarms/completions/document-analysis" \
     -H "Content-Type: application/json" \
     -d '{"task": "Analyze financial report", "priority": 5}'
```

## Monitoring and Logging 📊

SwarmDeploy provides built-in monitoring capabilities:

- Real-time task execution stats
- Error tracking and reporting
- Performance metrics
- Task history and audit logs

## Error Handling 🛡️

The system includes comprehensive error handling:

```python
try:
    result = await swarm.run(task)
except Exception as e:
    error_output = SwarmOutput(
        id=str(uuid.uuid4()),
        status="error",
        execution_time=time.time() - start_time,
        result=None,
        error=str(e)
    )
```

## Best Practices 🎯

1. Always set appropriate task priorities
2. Implement proper error handling
3. Use clustering for high-availability
4. Monitor system performance
5. Regular maintenance and updates

## Contributing 🤝

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Support 💬

- Email: kye@swarms.world
- Discord: [Join our community](https://swarms.world/swarms)
- Documentation: [https://docs.swarms.world](https://docs.swarms.world)

## License 📄

MIT License - see the [LICENSE](LICENSE) file for details.

---

Powered by [swarms.ai](https://swarms.ai) 🚀

For enterprise support and custom solutions, contact kye@swarms.world