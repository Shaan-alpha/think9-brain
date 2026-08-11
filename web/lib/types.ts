export type Outcome = "answered" | "refused" | "contested" | "routed";

export interface Citation {
  chunk_id: string;
  document_title: string;
  heading_path: string;
  deep_link: string;
  effective_date: string;
}

export interface Candidate {
  chunk_id: string;
  rank: number;
  score: number;
  snippet: string;
}

export interface Demotion {
  title: string;
  effective_date: string;
  demoted_by: string | null;
}

export interface ClaimVerdict {
  claim: string;
  supported: boolean;
  reason: string;
}

export interface Trace {
  route?: string;
  coverage?: number;
  owner?: string | null;
  scope?: [string, string];
  draft?: string;
  retrieval?: {
    dense?: Candidate[];
    sparse?: Candidate[];
    fused?: Candidate[];
    reranked?: Candidate[];
    demoted?: Demotion[];
  };
  verifier?: {
    stripped?: string[];
    claims?: ClaimVerdict[];
  };
  contested?: {
    attribute: string;
    values: [string, string][];
  };
}

export interface AskResponse {
  answer: string;
  outcome: Outcome;
  as_of: string | null;
  citations: Citation[];
  trace: Trace;
}
