from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import get_args, get_origin, get_type_hints


class Model:
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        hints = get_type_hints(cls)

        def convert(kind, value):
            if get_origin(kind) is list:
                return [convert(get_args(kind)[0], item) for item in value]
            if isinstance(kind, type) and issubclass(kind, Model):
                return kind.from_dict(value)
            return value

        return cls(**{key: convert(hints[key], value) for key, value in data.items()})


@dataclass
class StoryInfo(Model):
    story_id: str = ""


@dataclass
class GenerateInputCommon(Model):
    episode_nums: int = 0
    min_word_count_per_episode: int = 0
    max_word_count_per_episode: int = 0


@dataclass
class GenerateInputByCreativity(Model):
    common: GenerateInputCommon = field(default_factory=GenerateInputCommon)
    core_story: str = ""
    reference: str = ""
    role_setting: str = ""
    topic: str = ""
    world_view: str = ""


@dataclass
class StoryOutline(Model):
    role_info: list[str] = field(default_factory=list)
    story_outline: list[str] = field(default_factory=list)
    world_building: list[str] = field(default_factory=list)


@dataclass
class Chapter(Model):
    chapter_num: int = 0
    content: str = ""


@dataclass
class PlotPlanning(Model):
    description: str = ""
    chapter_from: int = 0
    chapter_to: int = 0
    chapters: list[Chapter] = field(default_factory=list)
    plot_id: int = 0
    title: str = ""


@dataclass
class PointPlanning(Model):
    description: str = ""


@dataclass
class SingleScriptProposal(Model):
    plot_planning: list[PlotPlanning] = field(default_factory=list)
    point_planning: PointPlanning = field(default_factory=PointPlanning)


@dataclass
class ScriptProposal(Model):
    script_proposals: list[SingleScriptProposal] = field(default_factory=list)


@dataclass
class OutlineEpisode(Model):
    content: str = ""
    episode_id: int = 0
    season_id: int = 0


@dataclass
class OutlineSeason(Model):
    episodes: list[OutlineEpisode] = field(default_factory=list)
    season_id: int = 0


@dataclass
class EpisodeOutline(Model):
    seasons: list[OutlineSeason] = field(default_factory=list)


@dataclass
class DramaEpisode(Model):
    episode_id: int = 0
    script_content: str = ""
    season_id: int = 0


@dataclass
class DramaSeason(Model):
    episodes: list[DramaEpisode] = field(default_factory=list)
    season_id: int = 0


@dataclass
class Drama(Model):
    seasons: list[DramaSeason] = field(default_factory=list)


@dataclass
class GenerateDrama(Model):
    episode_outline: EpisodeOutline = field(default_factory=EpisodeOutline)
    story_outline: StoryOutline = field(default_factory=StoryOutline)


@dataclass
class GeneratedData(Model):
    story_outline: StoryOutline = field(default_factory=StoryOutline)


@dataclass
class GenerateStoryOutline(Model):
    script_proposal: ScriptProposal = field(default_factory=ScriptProposal)


@dataclass
class GenerateScriptProposalByCreativityReq(Model):
    generate_input: GenerateInputByCreativity = field(
        default_factory=GenerateInputByCreativity
    )


@dataclass
class GenerateStoryOutlineByCreativityReq(Model):
    generate_data: GenerateStoryOutline = field(default_factory=GenerateStoryOutline)
    generate_input: GenerateInputByCreativity = field(
        default_factory=GenerateInputByCreativity
    )


@dataclass
class GenerateEpisodeOutlineByCreativityReq(Model):
    generate_input: GenerateInputByCreativity = field(
        default_factory=GenerateInputByCreativity
    )
    generated_data: GeneratedData = field(default_factory=GeneratedData)


@dataclass
class GenerateDramaByCreativityReq(Model):
    generate_data: GenerateDrama = field(default_factory=GenerateDrama)
    generate_input: GenerateInputByCreativity = field(
        default_factory=GenerateInputByCreativity
    )
    story_info: StoryInfo = field(default_factory=StoryInfo)


@dataclass
class GenerateScriptProposalRsp(Model):
    script_proposal: ScriptProposal = field(default_factory=ScriptProposal)
    story_outline: StoryOutline = field(default_factory=StoryOutline)


@dataclass
class GenerateStoryOutlineRsp(Model):
    story_outline: StoryOutline = field(default_factory=StoryOutline)


@dataclass
class GenerateEpisodeOutlineRsp(Model):
    episode_outline: EpisodeOutline = field(default_factory=EpisodeOutline)


@dataclass
class GenerateDramaRsp(Model):
    result: Drama = field(default_factory=Drama)
    episode_outline: EpisodeOutline = field(default_factory=EpisodeOutline)


SingleScriptProposal.PlotPlanning = PlotPlanning
SingleScriptProposal.PointPlanning = PointPlanning
PlotPlanning.Chapter = Chapter
EpisodeOutline.Episode = OutlineEpisode
EpisodeOutline.Season = OutlineSeason
Drama.Episode = DramaEpisode
Drama.Season = DramaSeason
