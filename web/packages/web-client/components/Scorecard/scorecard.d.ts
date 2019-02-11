export type PR = {
  assignees: string[];
  html_url: string;
  id: string;
  repo: string;
  state: string;
  status: string;
  title: string;
  user: string;
};

export type Issue = {
  assignees: string[];
  created_at: string;
  html_url: string;
  id: string;
  repo: string;
  title: string;
  urgent: boolean;
};
