import os
import json
import operator
import logging
from typing import TypedDict, Annotated, Sequence, Type, Optional, List, Dict, Any
from pydantic import BaseModel, Field

from langchain.tools.base import BaseTool
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_function

from dotenv import load_dotenv

# 設置 logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 加載環境變數
load_dotenv()

# 工具的輸入模型
class GetExchangeRateInput(BaseModel):
    """匯率查詢工具的輸入模型"""
    currency_pair: str = Field(description="要查詢匯率的貨幣 pair，例如 'USD/TWD'")

class GetAverageExchangeRateInput(BaseModel):
    """平均匯率查詢工具的輸入模型"""
    currency_pair: str = Field(description="要查詢匯率的貨幣 pair，例如 'USD/TWD'")
    period: str = Field(default="1 month", description="計算平均匯率的時間 period，例如 '1 month' 或 '3 months'")

class CalculateInput(BaseModel):
    """計算工具的輸入模型"""
    expression: str = Field(description="要計算的數學表達式，例如 '(31.5 - 31.0) / 31.0 * 100'")

# 工具實現
class GetExchangeRateTool(BaseTool):
    """工具：獲取當前匯率"""
    name: str = "get_exchange_rate"
    description: str = "取得指定貨幣對的當前匯率。"
    args_schema: Type[BaseModel] = GetExchangeRateInput

    def _run(self, currency_pair: str) -> str:
        """執行工具邏輯"""
        try:
            if not currency_pair:
                raise ValueError("貨幣 pair 不能為空。")
            
            exchange_rates = {
                "USD/TWD": 31.5,
                "EUR/TWD": 34.2,
                "JPY/TWD": 0.21,
            }
            rate = exchange_rates.get(currency_pair.strip(), None)
            if rate:
                return f"The current exchange rate for {currency_pair} is {rate}"
            else:
                return f"Exchange rate for {currency_pair} is not available"
        except Exception as e:
            logger.error(f"Error in GetExchangeRateTool: {str(e)}")
            raise

    async def _arun(self, currency_pair: str) -> str:
        """異步執行（未實現）"""
        raise NotImplementedError("異步執行尚未實現")

class GetAverageExchangeRateTool(BaseTool):
    """工具：獲取平均匯率"""
    name: str = "get_average_exchange_rate"
    description: str = "取得指定貨幣 pair 在特定時間段內的平均匯率。"
    args_schema: Type[BaseModel] = GetAverageExchangeRateInput

    def _run(self, currency_pair: str, period: str = "1 month") -> str:
        """執行工具邏輯"""
        try:
            if not currency_pair:
                raise ValueError("貨幣 pair 不能為空。")
            if not period:
                raise ValueError("時間 period 不能為空。")

            average_rates = {
                ("USD/TWD", "1 month"): 31.0,
                ("USD/TWD", "3 months"): 30.8,
                ("USD/TWD", "6 months"): 30.5,
                ("EUR/TWD", "1 month"): 33.8,
                ("EUR/TWD", "3 months"): 33.5,
                ("JPY/TWD", "1 month"): 0.208,
            }
            rate = average_rates.get((currency_pair.strip(), period.strip()), None)
            if rate:
                return f"The average exchange rate for {currency_pair} over {period} is {rate}"
            else:
                return f"Average exchange rate for {currency_pair} over {period} is not available"
        except Exception as e:
            logger.error(f"Error in GetAverageExchangeRateTool: {str(e)}")
            raise

    async def _arun(self, currency_pair: str, period: str = "1 month") -> str:
        """異步執行（未實現）"""
        raise NotImplementedError("異步執行尚未實現")

class CalculateTool(BaseTool):
    """工具：執行數學計算"""
    name: str = "calculate"
    description: str = "根據提供的數學表達式執行計算。例如：'(31.5 - 31.0) / 31.0 * 100' 計算漲幅百分比。"
    args_schema: Type[BaseModel] = CalculateInput

    def _run(self, expression: str) -> str:
        """執行工具邏輯"""
        try:
            if not expression:
                raise ValueError("表達式不能為空。")
            
            allowed_names = {"__builtins__": None}
            result = eval(expression.strip(), allowed_names, {})
            return f"計算結果為: {result}"
        except Exception as e:
            logger.error(f"Error in CalculateTool: {str(e)}")
            raise

    async def _arun(self, expression: str) -> str:
        """異步執行（未實現）"""
        raise NotImplementedError("異步執行尚未實現")

class AgentState(TypedDict):
    """代理狀態類型定義"""
    messages: Annotated[Sequence[BaseMessage], operator.add]

class Agent:
    """LLM Agent class"""
    
    def __init__(self, model: ChatOpenAI, tools: List[BaseTool], system_prompt: str = ""):
        """初始化 Agent"""
        try:
            logger.debug("Initializing Agent...")
            self.system_prompt = system_prompt
            self.tools: Dict[str, BaseTool] = {t.name: t for t in tools}
            
            # 檢查工具配置
            for tool in tools:
                logger.debug(f"Tool config - Name: {tool.name}")
                logger.debug(f"Tool config - Schema: {tool.args_schema if hasattr(tool, 'args_schema') else 'No schema'}")
            
            logger.debug("Setting up ToolExecutor...")
            self.tool_executor = ToolExecutor(tools)
            
            logger.debug("Converting tools to OpenAI functions...")
            openai_functions = []
            for tool in tools:
                try:
                    func = convert_to_openai_function(tool)
                    logger.debug(f"Converted function: {json.dumps(func, indent=2)}")
                    openai_functions.append(func)
                except Exception as e:
                    logger.error(f"Error converting tool {tool.name} to OpenAI function: {str(e)}")
                    raise
            
            logger.debug("Binding functions to model...")
            self.model = model.bind_functions(openai_functions)
            
            self.messages: List[BaseMessage] = []
            logger.debug("Agent initialization completed")
            self.setup_graph()
            
        except Exception as e:
            logger.error(f"Error during Agent initialization: {str(e)}")
            raise

    def setup_graph(self) -> None:
        """設置工作流程圖"""
        try:
            logger.debug("Setting up workflow graph...")
            graph = StateGraph(AgentState)
            graph.add_node("llm", self.call_model)
            graph.add_node("action", self.take_action)
            graph.add_conditional_edges(
                "llm",
                self.should_continue,
                {True: "action", False: END}
            )
            graph.add_edge("action", "llm")
            graph.set_entry_point("llm")
            self.app = graph.compile()
            logger.debug("Graph setup completed")
        except Exception as e:
            logger.error(f"Error in setup_graph: {str(e)}")
            raise

    def should_continue(self, state: AgentState) -> bool:
        """判斷是否繼續執行"""
        try:
            last_message = state['messages'][-1]
            
            # 如果不是 AI Message，或者是問題，停止
            if not isinstance(last_message, AIMessage):
                return False
                
            # 如果有函數調用，繼續
            if "function_call" in last_message.additional_kwargs:
                return True
                
            # 如果是一般回覆且不是問題，停止
            return False
        except Exception as e:
            logger.error(f"Error in should_continue: {str(e)}")
            return False

    def call_model(self, state: AgentState) -> AgentState:
        """調用模型獲取回應"""
        try:
            logger.debug("Calling model...")
            response = self.model.invoke(state['messages'])
            logger.debug(f"Model response received: {response}")
            return {'messages': state['messages'] + [response]}
        except Exception as e:
            logger.error(f"Error in call_model: {str(e)}")
            raise

    def take_action(self, state: AgentState) -> AgentState:
        """執行工具調用"""
        try:
            last_message = state['messages'][-1]
            function_call = last_message.additional_kwargs.get("function_call", {})
            
            logger.debug(f"Function call details: {json.dumps(function_call, indent=2)}")
            
            tool_name = function_call.get("name")
            tool_args_str = function_call.get("arguments", "{}")
            
            logger.debug(f"Parsing tool arguments: {tool_args_str}")
            tool_args = json.loads(tool_args_str)
            
            if tool_name not in self.tools:
                logger.error(f"Tool '{tool_name}' not found in available tools")
                result = f"Tool '{tool_name}' is not available."
            else:
                logger.debug(f"Executing tool {tool_name} with args: {tool_args}")
                action = ToolInvocation(
                    tool=tool_name,
                    tool_input=tool_args,
                )
                result = self.tool_executor.invoke(action)
                logger.debug(f"Tool execution result: {result}")

            return {'messages': state['messages'] + [FunctionMessage(content=str(result), name=tool_name)]}

        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {str(e)}")
            error_message = f"Error parsing tool arguments: {str(e)}"
            return {'messages': state['messages'] + [FunctionMessage(content=error_message, name="error")]}
        except Exception as e:
            logger.error(f"Error in take_action: {str(e)}")
            error_message = f"Error executing tool: {str(e)}"
            return {'messages': state['messages'] + [FunctionMessage(content=error_message, name="error")]}

    def process_turn(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """處理單次對話輪次"""
        try:
            logger.debug("Starting new turn processing...")
            logger.debug(f"Input messages: {[msg.content for msg in messages]}")
            
            state = {'messages': messages}
            final_state = self.app.invoke(state)
            
            logger.debug(f"Turn completed. Final messages: {[msg.content for msg in final_state['messages']]}")
            return final_state['messages']
            
        except Exception as e:
            logger.error(f"Error in process_turn: {str(e)}", exc_info=True)
            return messages + [AIMessage(content=f"抱歉，處理您的請求時發生錯誤: {str(e)}。請重試或換個方式提問。")]

    def run(self) -> None:
        """運行對話系統"""
        try:
            # 初始化 Message List
            if self.system_prompt:
                self.messages = [SystemMessage(content=self.system_prompt)]
            else:
                self.messages = []

            print("助手已準備就緒，請輸入您的問題（輸入 'quit' 結束對話）：")

            while True:
                try:
                    # 獲取 User 輸入
                    user_input = input("\nUser: ").strip()
                    
                    # 檢查是否退出
                    if user_input.lower() == 'quit':
                        print("\n感謝使用！再見！")
                        break
                    
                    # 如果輸入為空，繼續下一輪
                    if not user_input:
                        print("請輸入您的問題。")
                        continue

                    # 處理當前輪次
                    current_messages = self.messages + [HumanMessage(content=user_input)]
                    new_messages = self.process_turn(current_messages)
                    
                    # 更新 Message 歷史
                    self.messages = new_messages
                    
                    # Print New Message
                    for msg in new_messages[len(current_messages):]:
                        if isinstance(msg, FunctionMessage):
                            print(f"\nFunction '{msg.name}' output: {msg.content}")
                        elif isinstance(msg, AIMessage):
                            print(f"\nAssistant: {msg.content}")

                except KeyboardInterrupt:
                    print("\n\n對話被中斷。")
                    break
                except Exception as e:
                    logger.error(f"Error in conversation loop: {str(e)}", exc_info=True)
                    print(f"\n處理 Message 時發生錯誤: {str(e)}")
                    print("請重試或換個方式提問。")
                    continue

        except Exception as e:
            logger.error(f"Error in run method: {str(e)}", exc_info=True)
            print(f"\n運行錯誤: {str(e)}")
        finally:
            print("\n結束。")

    def get_conversation_history(self) -> List[str]:
        """獲取對話歷史"""
        history = []
        for msg in self.messages:
            if isinstance(msg, FunctionMessage):
                history.append(f"Function '{msg.name}': {msg.content}")
            elif isinstance(msg, HumanMessage):
                history.append(f"User: {msg.content}")


class ExchangeToolkit:
    """匯率 Toolkit"""
    
    def __init__(self):
        """初始化 Toolkit"""
        self.tools: List[BaseTool] = [
            GetExchangeRateTool(),
            GetAverageExchangeRateTool(),
            CalculateTool(),
        ]

    def get_tools(self) -> List[BaseTool]:
        """獲取 tool list"""
        return self.tools

def main():
    """Main function"""
    try:
        logger.info("Starting the application...")
        
        # 初始化 tool kit 和模型
        exchange_toolkit = ExchangeToolkit()
        tools = exchange_toolkit.get_tools()
        logger.debug(f"Initialized {len(tools)} tools")

        # 檢查環境變數
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("Missing OPENAI_API_KEY environment variable")

        # 建立 ChatOpenAI 模型 instance
        logger.debug("Creating ChatOpenAI model instance...")
        model = ChatOpenAI(
            model="gpt-4-turbo-preview",
            api_key=os.environ["OPENAI_API_KEY"],
            temperature=1,
        )

        # 定義系統提示
        system_prompt = """
        你是一名專業的外匯投資顧問助手，專門幫助投資者做出明智的匯率換匯決策。

        你可以：
        1. 查詢即時匯率 （使用 get_exchange_rate 工具）
        2. 分析歷史匯率趨勢 （使用 get_average_exchange_rate 工具）
        3. 進行相關數學計算（使用 calculate 工具來計算漲跌幅、報酬率等）
        4. 綜合各項指標提供換匯建議

        在分析時，你應該：
        1. 先獲取當前匯率
        2. 查詢歷史平均匯率作為參考
        3. 計算重要指標，例如：
            - 當前匯率相對於平均值的漲跌幅
            - 可能的獲利空間或風險
            - 相對報酬率

        請使用可用的工具來幫助用戶做出決策，並給出清晰的解釋和建議。
        """

        # 創建 Agent instance 並運行
        logger.info("Creating and starting Agent...")
        agent = Agent(model=model, tools=tools, system_prompt=system_prompt)
        agent.run()

    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        print(f"\n程式執行時發生錯誤: {str(e)}")
    finally:
        logger.info("Application terminated")
        print("\n程式結束。")

if __name__ == "__main__":
    main()