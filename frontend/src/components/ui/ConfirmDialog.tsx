import Modal from "./Modal";
import Button from "./Button";

interface Props {
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
  confirmLabel?: string;
}

export default function ConfirmDialog({ message, onConfirm, onCancel, confirmLabel = "Delete" }: Props) {
  return (
    <Modal
      open
      onClose={onCancel}
      size="sm"
      footer={
        <>
          <Button variant="secondary" onClick={onCancel}>Cancel</Button>
          <Button variant="danger" onClick={onConfirm}>{confirmLabel}</Button>
        </>
      }
    >
      <p className="text-sm">{message}</p>
    </Modal>
  );
}
