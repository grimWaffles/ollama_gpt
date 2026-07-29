from tools.file_operation_tool import FolderReadTool, FolderWriteTool
from tools.web_search_tool import WebSearchTool


class SystemToolService:
    def __init__(self):
        self._reader = FolderReadTool()
        self._writer = FolderWriteTool()
        self._web_search = WebSearchTool()

    def get_system_tools(self):
        return [self._reader, self._writer, self._web_search]