import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  render() {
    if (this.state.error) {
      return (
        <div className="flex h-screen items-center justify-center bg-nova-bg p-8">
          <div className="max-w-md rounded-lg border border-nova-error/30 bg-nova-surface p-6 text-center">
            <h1 className="mb-2 text-lg font-semibold text-nova-error">Something went wrong</h1>
            <p className="mb-4 text-sm text-nova-text-dim">{this.state.error.message}</p>
            <button
              onClick={() => this.setState({ error: null })}
              className="rounded bg-nova-accent px-4 py-2 text-sm font-medium text-white hover:bg-nova-accent-hover"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
