import ChatPage from "../pages/ChatPage";

/** Interrogate Nova — the full chat, docked in-world. `embedded` drops the
 *  legacy 280px conversation sidebar so the cosmos stays chrome-free. */
export default function ChatPanel() {
  return (
    <div className="h-full">
      <ChatPage embedded />
    </div>
  );
}
