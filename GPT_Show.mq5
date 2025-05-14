//+------------------------------------------------------------------+
//|                                                      GPT_Show.mq5|
//|           Draw GPT trade signals on chart from JSONL log         |
//+------------------------------------------------------------------+
#property script_show_inputs

input string LogFileName = "gpt_signals_log.jsonl";

void OnStart()
{
   ResetLastError();
   int fileHandle = FileOpen(LogFileName, FILE_READ | FILE_TXT | FILE_ANSI);
   if(fileHandle == INVALID_HANDLE)
   {
      Print("Failed to open file: ", LogFileName);
      return;
   }

   int i = 0;
   while(!FileIsEnding(fileHandle))
   {
      string line = FileReadString(fileHandle);
      if(StringFind(line, ""entry":") != -1)
      {
         double entry = StringToDouble(ReadJsonValue(line, "entry"));
         string symbol = ReadJsonValue(line, "symbol");
         string signal = ReadJsonValue(line, "signal");
         string timeStr = ReadJsonValue(line, "timestamp");
         datetime t = StringToTime(timeStr);

         string name = "GPT_"+IntegerToString(i);
         int arrowCode = signal == "SELL" ? 234 : 233;
         color col = signal == "SELL" ? clrRed : clrLime;

         ObjectCreate(0, name, OBJ_ARROW, 0, t, entry);
         ObjectSetInteger(0, name, OBJPROP_ARROWCODE, arrowCode);
         ObjectSetInteger(0, name, OBJPROP_COLOR, col);
         ObjectSetInteger(0, name, OBJPROP_WIDTH, 2);
      }
      i++;
   }

   FileClose(fileHandle);
}

string ReadJsonValue(string line, string key)
{
   int p = StringFind(line, """ + key + "":");
   if(p == -1) return "";
   int start = StringFind(line, ":", p) + 1;
   int end = StringFind(line, ",", start);
   if(end == -1) end = StringLen(line) - 1;
   string value = StringSubstr(line, start, end - start);
   value = StringTrim(value);
   value = StringReplace(value, "\"", "");
   value = StringReplace(value, """, "");
   return value;
}