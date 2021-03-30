from azureml.core import Workspace

subscription_id = '63584c7c-979a-40e5-a3c9-93913eda5fbd'
resource_group  = 'mlresourcegr'
workspace_name  = 'mlworkspace'

try:
    ws = Workspace(subscription_id = subscription_id, resource_group = resource_group, workspace_name = workspace_name)
    ws.write_config()
    print('Library configuration succeeded')
except:
    print('Workspace not found')